#!/usr/bin/env python3
"""
Performance comparison: PyTorch copy_() vs SYCL transfer_mamba_state kernel.

Tests all three HiCache transfer directions:
  - Device → Device (L1 internal)
  - Device → Host (backup/write L1→L2)
  - Host → Device (restore/read L2→L1)

Usage:
  python tests/perf_mamba_transfer.py
  python tests/perf_mamba_transfer.py --sizes 1048576 3145728 8388608
  python tests/perf_mamba_transfer.py --num-tokens 32 --warmup 3 --iters 10
"""
import argparse
import time
import torch

# Default sizes to test (bytes)
DEFAULT_SIZES = [
    131072,    # 128 KB
    524288,    # 512 KB
    1048576,   # 1 MB (Qwen3.6-35B Mamba)
    3145728,   # 3 MB
    8388608,   # 8 MB
    16777216,  # 16 MB (TIER2_LIMIT)
]


def pytorch_copy(src, dst, src_indices, dst_indices):
    """PyTorch copy_() baseline."""
    src_idx_cpu = src_indices.cpu().tolist()
    dst_idx_cpu = dst_indices.cpu().tolist()
    for si, di in zip(src_idx_cpu, dst_idx_cpu):
        dst[di].copy_(src[si], non_blocking=False)
    torch.xpu.synchronize()


def sycl_kernel_copy(src, dst, src_indices, dst_indices, item_size):
    """SYCL transfer_mamba_state kernel."""
    from sgl_kernel.kvcacheio import transfer_mamba_state
    transfer_mamba_state(src, dst, src_indices, dst_indices, item_size=item_size)
    torch.xpu.synchronize()


def benchmark_one(name, fn, warmup=3, iters=10):
    """Run warmup + timed iterations, return mean ms."""
    for _ in range(warmup):
        fn()

    torch.xpu.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)

    return sum(times) / len(times)


def run_benchmark(
    item_size_bytes: int,
    num_tokens: int,
    direction: str,
    warmup: int = 3,
    iters: int = 10,
    dtype=torch.bfloat16,
):
    """
    Benchmark one configuration.

    direction: 'd2d' (device→device), 'd2h' (device→host), 'h2d' (host→device)
    """
    # Clear cache before each benchmark to avoid OOM
    torch.xpu.empty_cache()

    # Use smaller pool for large items to avoid OOM
    pool_size = min(128, max(64, num_tokens * 2))
    item_size_elements = item_size_bytes // dtype.itemsize

    # Create tensors based on direction
    if direction == 'd2d':
        src = torch.randn(pool_size, item_size_elements, dtype=dtype, device='xpu')
        dst_pytorch = torch.zeros_like(src)
        dst_sycl = torch.zeros_like(src)
    elif direction == 'd2h':
        src = torch.randn(pool_size, item_size_elements, dtype=dtype, device='xpu')
        dst_pytorch = torch.zeros(pool_size, item_size_elements, dtype=dtype).pin_memory()
        dst_sycl = torch.zeros(pool_size, item_size_elements, dtype=dtype).pin_memory()
    elif direction == 'h2d':
        src = torch.randn(pool_size, item_size_elements, dtype=dtype).pin_memory()
        dst_pytorch = torch.zeros(pool_size, item_size_elements, dtype=dtype, device='xpu')
        dst_sycl = torch.zeros(pool_size, item_size_elements, dtype=dtype, device='xpu')
    else:
        raise ValueError(f"Unknown direction: {direction}")

    indices = torch.randperm(pool_size, dtype=torch.int64, device='xpu')[:num_tokens]

    # Benchmark PyTorch copy_()
    pytorch_ms = benchmark_one(
        "pytorch",
        lambda: pytorch_copy(src, dst_pytorch, indices, indices),
        warmup=warmup,
        iters=iters,
    )

    # Benchmark SYCL kernel
    try:
        sycl_ms = benchmark_one(
            "sycl",
            lambda: sycl_kernel_copy(src, dst_sycl, indices, indices, item_size_bytes),
            warmup=warmup,
            iters=iters,
        )
    except Exception as e:
        sycl_ms = None
        sycl_error = str(e)

    # Verify correctness
    if sycl_ms is not None:
        if direction == 'd2h':
            correct = torch.allclose(dst_pytorch, dst_sycl)
        elif direction == 'h2d':
            correct = torch.allclose(dst_pytorch, dst_sycl)
        else:
            correct = torch.allclose(dst_pytorch, dst_sycl)
    else:
        correct = None

    result = {
        'item_size_mb': item_size_bytes / 1024 / 1024,
        'num_tokens': num_tokens,
        'direction': direction,
        'total_mb': item_size_bytes * num_tokens / 1024 / 1024,
        'pytorch_ms': pytorch_ms,
        'sycl_ms': sycl_ms,
        'speedup': pytorch_ms / sycl_ms if sycl_ms else None,
        'correct': correct,
    }

    # Cleanup to avoid OOM
    del src, dst_pytorch, dst_sycl, indices
    torch.xpu.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mamba state transfer")
    parser.add_argument('--sizes', type=int, nargs='+', default=DEFAULT_SIZES,
                        help='Item sizes in bytes to test')
    parser.add_argument('--num-tokens', type=int, default=16,
                        help='Number of tokens to transfer')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=10, help='Timed iterations')
    parser.add_argument('--directions', type=str, nargs='+',
                        default=['d2d', 'd2h', 'h2d'],
                        help='Transfer directions to test')
    args = parser.parse_args()

    print(f"Mamba State Transfer Benchmark")
    print(f"  num_tokens={args.num_tokens}, warmup={args.warmup}, iters={args.iters}")
    print(f"  directions={args.directions}")
    print()

    results = []
    for direction in args.directions:
        dir_name = {'d2d': 'Device→Device', 'd2h': 'Device→Host (backup)',
                    'h2d': 'Host→Device (restore)'}[direction]
        print(f"=== {dir_name} ===")
        print(f"{'Size':>8} {'Total':>8} {'PyTorch':>10} {'SYCL':>10} {'Speedup':>8} {'OK':>4}")
        print("-" * 60)

        for size in args.sizes:
            r = run_benchmark(
                item_size_bytes=size,
                num_tokens=args.num_tokens,
                direction=direction,
                warmup=args.warmup,
                iters=args.iters,
            )
            results.append(r)

            size_str = f"{r['item_size_mb']:.1f}MB"
            total_str = f"{r['total_mb']:.1f}MB"
            pytorch_str = f"{r['pytorch_ms']:.2f}ms"
            sycl_str = f"{r['sycl_ms']:.2f}ms" if r['sycl_ms'] else "FAIL"
            speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
            ok_str = "✓" if r['correct'] else "✗" if r['correct'] is False else "?"

            print(f"{size_str:>8} {total_str:>8} {pytorch_str:>10} {sycl_str:>10} {speedup_str:>8} {ok_str:>4}")

        print()

    # Summary
    print("=== Summary ===")
    wins = sum(1 for r in results if r['speedup'] and r['speedup'] > 1.0)
    losses = sum(1 for r in results if r['speedup'] and r['speedup'] < 1.0)
    fails = sum(1 for r in results if r['sycl_ms'] is None)
    print(f"SYCL wins: {wins}, PyTorch wins: {losses}, SYCL fails: {fails}")

    if any(r['speedup'] for r in results):
        avg_speedup = sum(r['speedup'] for r in results if r['speedup']) / sum(1 for r in results if r['speedup'])
        print(f"Average speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
