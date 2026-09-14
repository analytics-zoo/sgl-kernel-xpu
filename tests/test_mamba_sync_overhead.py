#!/usr/bin/env python3
"""
Test sync overhead: SYCL kernel vs PyTorch with different sync strategies.

Compares:
1. SYCL kernel (single dispatch)
2. PyTorch sync-at-end (best case, but risks BCS timeout)
3. PyTorch adaptive sync (~32 MB/sync, safe fallback)
"""
import time
import torch

def pytorch_sync_at_end(src, dst, indices, item_size):
    """PyTorch copy with sync only at end - fastest but may hit BCS watchdog."""
    idx_list = indices.cpu().tolist()
    for i in idx_list:
        dst[i].copy_(src[i], non_blocking=False)
    torch.xpu.synchronize()


def pytorch_adaptive_sync(src, dst, indices, item_size):
    """PyTorch copy with adaptive sync - safe fallback."""
    SYNC_TARGET_BYTES = 32 * 1024 * 1024  # 32 MB
    sync_batch = max(1, SYNC_TARGET_BYTES // item_size)
    idx_list = indices.cpu().tolist()
    for i, idx in enumerate(idx_list):
        dst[idx].copy_(src[idx], non_blocking=False)
        if (i + 1) % sync_batch == 0:
            torch.xpu.synchronize()
    torch.xpu.synchronize()  # Final sync


def sycl_kernel(src, dst, indices, item_size):
    """SYCL transfer_mamba_state kernel."""
    from sgl_kernel.kvcacheio import transfer_mamba_state
    transfer_mamba_state(src, dst, indices, indices, item_size=item_size)
    torch.xpu.synchronize()


def benchmark(name, fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def run_test(item_size_bytes, num_tokens, direction='h2d'):
    """Run comparison for one configuration."""
    torch.xpu.empty_cache()

    pool_size = max(64, num_tokens * 2)
    item_elements = item_size_bytes // 2  # bfloat16

    if direction == 'h2d':
        src = torch.randn(pool_size, item_elements, dtype=torch.bfloat16).pin_memory()
        dst = torch.zeros(pool_size, item_elements, dtype=torch.bfloat16, device='xpu')
    elif direction == 'd2h':
        src = torch.randn(pool_size, item_elements, dtype=torch.bfloat16, device='xpu')
        dst = torch.zeros(pool_size, item_elements, dtype=torch.bfloat16).pin_memory()
    else:  # d2d
        src = torch.randn(pool_size, item_elements, dtype=torch.bfloat16, device='xpu')
        dst = torch.zeros_like(src)

    indices = torch.randperm(pool_size, dtype=torch.int64, device='xpu')[:num_tokens]

    # Calculate expected sync batch
    sync_batch = max(1, (32 * 1024 * 1024) // item_size_bytes)

    sycl_ms = benchmark("sycl", lambda: sycl_kernel(src, dst, indices, item_size_bytes))
    end_ms = benchmark("end", lambda: pytorch_sync_at_end(src, dst, indices, item_size_bytes))
    adaptive_ms = benchmark("adaptive", lambda: pytorch_adaptive_sync(src, dst, indices, item_size_bytes))

    return {
        'item_mb': item_size_bytes / 1024 / 1024,
        'tokens': num_tokens,
        'sync_batch': sync_batch,
        'sycl_ms': sycl_ms,
        'pytorch_end_ms': end_ms,
        'pytorch_adaptive_ms': adaptive_ms,
        'sycl_vs_end': end_ms / sycl_ms,
        'sycl_vs_adaptive': adaptive_ms / sycl_ms,
    }


def main():
    print("Mamba Transfer: SYCL vs PyTorch Sync Strategies")
    print("=" * 80)
    print()

    configs = [
        # (item_size_bytes, num_tokens)
        (524288, 16),    # 512 KB × 16 = 8 MB
        (1048576, 16),   # 1 MB × 16 = 16 MB  (Qwen3.6-35B)
        (1048576, 32),   # 1 MB × 32 = 32 MB
        (3145728, 16),   # 3 MB × 16 = 48 MB
    ]

    for direction in ['h2d', 'd2h']:
        dir_name = 'Host→Device (restore)' if direction == 'h2d' else 'Device→Host (backup)'
        print(f"=== {dir_name} ===")
        print(f"{'Size':>6} {'Tok':>4} {'Batch':>5} │ {'SYCL':>8} {'PT-End':>8} {'PT-Adap':>8} │ {'vs End':>7} {'vs Adap':>8}")
        print("-" * 80)

        for item_size, num_tokens in configs:
            r = run_test(item_size, num_tokens, direction)
            print(f"{r['item_mb']:>5.1f}M {r['tokens']:>4} {r['sync_batch']:>5} │ "
                  f"{r['sycl_ms']:>7.2f}ms {r['pytorch_end_ms']:>7.2f}ms {r['pytorch_adaptive_ms']:>7.2f}ms │ "
                  f"{r['sycl_vs_end']:>6.2f}x {r['sycl_vs_adaptive']:>7.2f}x")
        print()

    print("Legend:")
    print("  Batch = sync every N tokens for adaptive strategy")
    print("  vs End = PyTorch(sync-at-end) / SYCL  (>1 = SYCL faster)")
    print("  vs Adap = PyTorch(adaptive) / SYCL   (>1 = SYCL faster)")


if __name__ == "__main__":
    main()
