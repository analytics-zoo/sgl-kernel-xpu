# sgl-kernel-xpu — Kernel Inventory

Intel XPU fork of `sglang/sgl-kernel` — the native SYCL / cutlass-sycl kernel
library behind `import sgl_kernel` on Intel GPUs. Upstream at
[sgl-project/sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu)
(current HEAD in this tree: `c668bb6`, branch `main`).

Python package name is `sgl_kernel` (the same identifier as the CUDA
sgl-kernel); the XPU version is a **drop-in replacement** — the same
`torch.ops.sgl_kernel.*` op surface, but the ops under the hood are SYCL
kernels that dispatch to `torch::kXPU`.

## Tech stack

| Layer            | Choice                                                                                |
|------------------|----------------------------------------------------------------------------------------|
| Build system     | **scikit-build-core + CMake (Ninja)** — `pyproject.toml` drives `cmake` via `scikit_build_core.build` |
| Target           | **Battlemage / Xe2 / BMG-G21** (`DPCPP_SYCL_TARGET = bmg`, AOT: `-device bmg`)         |
| Language         | SYCL (DPC++), C++20, `-fsycl -fsycl-targets=spir64_gen`                                |
| Matmul primitives| **cutlass-sycl** (vendored via `FetchContent` from `github.com/intel/sycl-tla` @ `2fc09973`) — headers-only mode; CUTLASS patterns for Xe2 DPAS |
| Compiler         | oneAPI **icpx 2025.3.3** / SYCL compiler ≥ `20250806` (build fails below)              |
| PyTorch          | 2.10 + xpu in the Dockerfile; 2.11+xpu also works (ABI compatible)                     |
| Python package   | `sgl_kernel`, version `0.1.8` (from `python/sgl_kernel/version.py`)                    |
| Extension ABI    | `wheel.py-api = "cp39"`, `USE_SABI` (stable ABI) — cross-Python compatible             |
| Kernel split     | Files ending in `Xe20.cpp` / `Xe20.sycl` compile **AOT** for BMG; others compile with the common (JIT / SPIR-V) path. Split is done in `src/CMakeLists.txt`. |
| Torch library    | `TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)` in `src/torch_extension_sycl.cc` (one namespace) |
| Python module    | `common_ops` (one `.so` per kernel source, all linked into `sgl_kernel.common_ops`)    |
| SPIR-V ext       | `+SPV_INTEL_split_barrier`, `+SPV_INTEL_2d_block_io`, `+SPV_INTEL_subgroup_matrix_multiply_accumulate` (required for Xe2 DPAS + 2D block IO) |
| Template library | CUTLASS-SYCL for attention / MLA / grouped-GEMM kernels (`kernels/flash_attention_v2`, `kernels/mla`, `kernels/chunk_prefill`, `kernels/moe/xe20`) |

### How install works

```bash
source /opt/intel/oneapi/setvars.sh
pip install -v .            # scikit-build-core invokes CMake, builds a wheel
# or for incremental builds:
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
  python -m uv build --wheel -Cbuild-dir=build --color=always .
```

CMake fetches cutlass-sycl on first configure (~minutes of download); subsequent
builds are incremental. The wheel drops `common_ops.so` + one `.so` per SYCL
source into `sgl_kernel/`, plus the Python wrappers in `python/sgl_kernel/`.

## Layout

```
sgl-kernel-xpu/
├── CMakeLists.txt                   # top-level: cutlass FetchContent, subdir src
├── pyproject.toml                   # scikit-build-core, sgl-kernel 0.11.0 dist, py-api=cp39
├── Dockerfile.xpu_kernel            # BMG image: intel/dle:2025.3.2 + torch 2.10.0 + this pkg
├── cmake/                           # SYCL/MKL/CCL finders, BuildFlags
├── src/
│   ├── torch_extension_sycl.cc      # TORCH_LIBRARY_FRAGMENT(sgl_kernel, ...)
│   ├── CMakeLists.txt               # Xe20 vs common split
│   ├── BuildOnLinux.cmake           # one .so per .cpp, all linked into common_ops
│   ├── FMHADecodeXe20.cmake         # per-subsystem CMake (cutlass instance gen)
│   ├── FMHAPrefillXe20.cmake
│   ├── MlaDecodeXe20.cmake
│   ├── GroupGemmXe20.cmake
│   └── sycl/                        # SYCL sources (common) + *Xe20.cpp (AOT)
│       ├── comm/                    #   General helpers, SLM copy, AccumulateType
│       ├── kernels/
│       │   ├── flash_attention_v2/  #   cutlass FMHA fwd (decode + prefill)
│       │   ├── chunk_prefill/       #   chunked prefill path
│       │   ├── mla/                 #   MLA decode (DeepSeek-style)
│       │   └── moe/xe20/            #   Grouped-GEMM for MoE
│       ├── flash_attention.cpp      # Xe20 MHA entry (decode / prefill / chunkprefill dispatch)
│       ├── mla_decode.cpp           # flash_mla_decode + workspace size
│       ├── GroupGemmXe20.cpp        # moe_grouped_mm_nt_xe20
│       ├── FusedQKNormRope.cpp      # fused QK RMSNorm + RoPE (YARN)
│       ├── RMSNorm.cpp / Rope.cpp   # elementwise ops
│       ├── MoE{Align,Sum,SumReduce,PrepareInputs,_fused_gate,_sum_reduce}.cpp
│       ├── TopKSoftMax.cpp          # router topk
│       ├── TripleOps.cpp            # silu/gelu_and_mul
│       ├── SwigluAlphaLimit.cpp     # swiglu_gpt_oss_sigmoid_alpha
│       ├── per_tensor_quant_fp8.cpp # per-tensor FP8 scale
│       ├── per_token_group_quant_8bit(_v2).cpp  # FP8 / INT8 per-group
│       ├── per_token_group_quant_fp4.cpp        # FP4 per-group
│       ├── awq_dequantize.cpp       # AWQ INT4 dequant
│       ├── merge_states.cpp         # attention state merge (for split-K / speculative)
│       ├── Device.cpp               # query_device (SM/Xe arch capability)
│       └── *.cpp.in                 # CMake-generated kernel instances (per shape)
├── include/
│   ├── sgl_kernel_ops.h             # public op decls
│   ├── sgl_flash_kernel_ops.h       # flash-attn-varlen forward decls (upstream shape)
│   └── sgl_kernel_torch_shim.h      # make_pytorch_shim() — int/double → int64_t/double
├── python/sgl_kernel/
│   ├── __init__.py                  # re-exports ~all ops under `sgl_kernel.*`
│   ├── attention.py                 # lightning_attention_decode, merge_state{,_v2}, flash_mla_decode
│   ├── elementwise.py               # rmsnorm / silu_and_mul / apply_rope / fused_qk_norm_rope
│   ├── gemm.py                      # awq_dequantize, fp8/int8 GEMM, quant helpers
│   ├── moe.py                       # moe_align_block_size, topk_softmax, moe_grouped_mm_nt, ...
│   ├── flash_attn.py                # mha_fwd wrapper (cu_seqlens varlen path)
│   ├── sampling.py                  # top-k/top-p/min-p sampling-from-probs
│   ├── speculative.py               # tree speculative sampling, verify_tree_greedy
│   ├── sparse_flash_attn.py         # sparse FMHA varlen
│   ├── allreduce.py                 # (stub; XPU custom allreduce not compiled here)
│   ├── grammar.py                   # xgrammar bitmask helpers
│   └── utils.py                     # get_device_capability, is_xe2_arch, get_xpu_stream
├── benchmark/                       # triton-testing style benchmarks
└── tests/                           # pytest suite (~50 test files)
```

## Kernel inventory (by domain)

All ops live under **`torch.ops.sgl_kernel.*`** (one shared namespace).
Python wrappers in `python/sgl_kernel/*.py` thinly forward with docstrings +
optional output-buffer handling.

### 1. Flash Attention (cutlass-sycl Xe20)

| Op / Python symbol                 | What                                                                      |
|------------------------------------|----------------------------------------------------------------------------|
| `fwd` → `flash_attn.mha_fwd`       | Varlen MHA forward: decode / prefill / chunk-prefill auto-dispatched by `flash_attention.cpp`. Supports paged KV, RoPE cos/sin fused in, GQA, softcap, descale (fp8 Q/K/V), sinks, rotary interleave, scheduler-metadata (split-KV) |
| `lightning_attention_decode`       | Lightning-style linear-attention decode (single-token)                     |
| `merge_state` / `merge_state_v2`   | Merge partial attention states (split-KV / hierarchical)                   |

Sources: `sycl/flash_attention.cpp`, `sycl/kernels/flash_attention_v2/` (cutlass
templates), `sycl/kernels/chunk_prefill/`, `sycl/merge_states.cpp`.

### 2. MLA decode (DeepSeek-V2/V3)

| Op                                 | What                                                                      |
|------------------------------------|----------------------------------------------------------------------------|
| `flash_mla_decode`                 | Multi-head latent attention decode (D_latent=512, D_rope=64)              |
| `flash_mla_get_workspace_size`     | Query workspace bytes for a (max_seq_len, num_batches, num_kv_splits)     |

Source: `sycl/mla_decode.cpp`, `sycl/kernels/mla/` (cutlass MLA templates).

### 3. RMSNorm / RoPE / elementwise

| Op                                  | What                                                                       |
|-------------------------------------|-----------------------------------------------------------------------------|
| `rmsnorm`                           | `out = (x / RMS(x)) * weight`                                               |
| `fused_add_rmsnorm`                 | `residual += x; x = rmsnorm(residual) * weight`                             |
| `gemma_rmsnorm` / `gemma_fused_add_rmsnorm` | Gemma `weight + 1.0` convention                                     |
| `rotary_embedding`                  | RoPE applied to Q/K (NeoX or interleaved)                                  |
| `apply_rope_pos_ids_cos_sin_cache`  | In-place RoPE with precomputed cos/sin cache (SGL/vLLM-compatible layout)  |
| `fused_qk_norm_rope`                | Q/K RMSNorm + RoPE in a single kernel (supports **YARN** scaling)          |
| `silu_and_mul` / `gelu_and_mul` / `gelu_tanh_and_mul` | gate-up activations                                      |
| `swiglu_gpt_oss_sigmoid_alpha`      | GPT-OSS swiglu variant (alpha/limit)                                       |

Sources: `sycl/RMSNorm.cpp`, `sycl/Rope.cpp`, `sycl/FusedQKNormRope.cpp`,
`sycl/TripleOps.cpp`, `sycl/SwigluAlphaLimit.cpp`.

### 4. Quantization

| Op                                    | What                                                                     |
|---------------------------------------|---------------------------------------------------------------------------|
| `sgl_per_tensor_quant_fp8`            | Per-tensor FP8 quant (static or dynamic)                                 |
| `sgl_per_token_quant_fp8`             | Per-token FP8 quant                                                      |
| `sgl_per_token_group_quant_8bit`      | Per-token + per-group FP8/INT8 (with `scale_ue8m0` flag)                 |
| `sgl_per_token_group_quant_8bit_v2`   | Same + `fuse_silu_and_mul` + `masked_m` for EP MoE                       |
| `sgl_per_token_group_quant_fp4`       | Per-group FP4 quant                                                      |
| `scaled_fp4_quant`                    | FP4 quant with per-block scale                                           |
| `awq_dequantize`                      | AWQ INT4 weight dequant                                                  |

Sources: `sycl/per_tensor_quant_fp8.cpp`, `sycl/per_token_group_quant_*.cpp`,
`sycl/awq_dequantize.cpp`.

### 5. GEMM / BMM

| Op                                | What                                                                         |
|-----------------------------------|-------------------------------------------------------------------------------|
| `fp8_scaled_mm`                   | FP8 GEMM with per-tensor or per-token scales                                  |
| `fp8_blockwise_scaled_mm`         | FP8 GEMM with blockwise scales                                                |
| `int8_scaled_mm`                  | INT8 GEMM with per-tensor / per-token scales                                  |
| `bmm_fp8`                         | Batched FP8 GEMM (needs workspace + sycl-stream passthrough)                  |
| `cutlass_scaled_fp4_mm`           | FP4 GEMM (cutlass-sycl backend)                                               |
| `qserve_w4a8_per_chn_gemm`        | QServe INT4-weight × INT8-activation per-channel                              |
| `qserve_w4a8_per_group_gemm`      | QServe INT4-weight × INT8-activation per-group                                |

Sources: `sycl/kernels/moe/xe20/` (cutlass templates are shared with grouped),
individual GEMM sources referenced through `include/sgl_kernel_ops.h`.

### 6. MoE path (end-to-end block building)

| Op                                    | What                                                                     |
|---------------------------------------|---------------------------------------------------------------------------|
| `moe_align_block_size`                | Align topk_ids into per-expert blocks + cumsum buffer                    |
| `moe_sum`                             | Per-token sum over experts                                               |
| `moe_sum_reduce`                      | Routed-scaling + sum-reduce                                              |
| `topk_softmax`                        | Fused softmax + top-K router                                             |
| `moe_fused_gate`                      | Fused gating with bias + group routing + shared-expert reduction         |
| `moe_grouped_mm_nt_xe20`              | **Grouped GEMM** N×T (silu/gelu/swiglu fuse); cutlass-sycl AOT for BMG   |
| `prepare_moe_input`                   | Build expert offsets, problem_sizes, input/output permutations           |
| `ep_moe_pre_reorder`                  | EP MoE scatter (per-token, optional per-token scale)                     |
| `ep_moe_silu_and_mul`                 | EP MoE activation stage                                                  |
| `ep_moe_post_reorder`                 | EP MoE gather / weighted sum                                             |
| `scatter_tokens_to_experts`           | Generic scatter using `src2dst_map`                                      |
| `apply_shuffle_mul_sum`               | Permute + routed-scale-multiply + sum-reduce                             |
| `fp8_blockwise_scaled_grouped_mm`     | FP8 blockwise grouped GEMM (CUTLASS path; used on upstream)              |
| `cutlass_fp4_group_mm`                | FP4 grouped GEMM                                                         |
| `scaled_fp4_experts_quant`            | Per-expert FP4 quant                                                     |

Sources: `sycl/MoEAlign.cpp`, `sycl/MoE_fused_gate.cpp`,
`sycl/MoEPrepareInputs.cpp`, `sycl/MoESum.cpp`, `sycl/MoE_sum_reduce.cpp`,
`sycl/TopKSoftMax.cpp`, `sycl/GroupGemmXe20.cpp`.

### 7. Sampling (FlashInfer-compatible API)

Namespace: `torch.ops.sgl_kernel.*`.

| Op                                       | What                                       |
|------------------------------------------|---------------------------------------------|
| `min_p_sampling_from_probs`              | Min-p sampling                              |
| `top_k_renorm_probs`                     | Top-k renormalization                       |
| `top_p_renorm_probs`                     | Top-p renormalization                       |
| `top_k_top_p_sampling_from_probs`        | Combined top-k / top-p sampling             |
| `top_p_sampling_from_probs`              | Top-p sampling                              |

### 8. Speculative decoding (tree-based)

| Op                                         | What                                                                  |
|--------------------------------------------|------------------------------------------------------------------------|
| `tree_speculative_sampling_target_only`    | Target-only probabilistic verification                                |
| `verify_tree_greedy`                       | Greedy tree verification                                              |
| `build_tree_kernel_efficient`              | Build draft tree (parent list + retrieve indices)                     |
| `segment_packbits`                         | Packbits for tree attention masks                                     |

Source: `sycl/` tree-verify sources referenced through `sgl_kernel_ops.h`.

### 9. Grammar (XGrammar)

| Op                            | What                                                        |
|-------------------------------|-------------------------------------------------------------|
| `ApplyTokenBitmaskInplace`    | In-place mask logits by allowed-token bitmask               |

### 10. Device / utility

| Op                          | What                                              |
|-----------------------------|---------------------------------------------------|
| `query_device`              | Returns `(major, minor)` compute capability — Xe2 (BMG-G21) reports major=2; drives `is_xe2_arch()` guards in `python/sgl_kernel/utils.py`. |

## Integration with sglang main

Production sglang imports from `sgl_kernel` for many XPU-or-generic paths.
Representative (non-exhaustive) consumers in the sglang source tree:

- `srt/layers/layernorm.py` — imports `rmsnorm, fused_add_rmsnorm, gemma_*` family.
- `srt/layers/quantization/fp8_utils.py` — `fp8_blockwise_scaled_mm, fp8_scaled_mm`.
- `srt/layers/quantization/w8a8_int8.py` — `int8_scaled_mm`.
- `srt/models/deepseek_common/attention_forward_methods/forward_mha.py` — `merge_state_v2` for split-KV.
- `srt/models/deepseek_v2.py`, `deepseek.py` — `dsv3_fused_a_gemm, dsv3_router_gemm` (CUDA-only in our fork; XPU fallback path).
- `srt/layers/moe/topk.py` — `moe_fused_gate, kimi_k2_moe_fused_gate`.
- `srt/models/bailing_moe_linear.py`, `longcat_flash.py` — `awq_dequantize`.

In short, most "common infra" kernels (layernorm, quant, GEMM, basic MoE, RoPE,
sampling, tree-spec) that sglang uses are backed by this package when running
on XPU.

## Relation to other kernel packages in this workspace

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ sglang (Python)                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  `import sgl_kernel`                                                         │
│  ├─ this repo (sgl-kernel-xpu)  ← pure SYCL + cutlass-sycl, Xe2-AOT          │
│  │    • attention, MLA decode, MoE grouped GEMM, quant, RoPE/RMSNorm,        │
│  │      sampling, speculative, grammar                                       │
│  │                                                                           │
│  optional plugins (not part of sgl-kernel, different packages):              │
│  ├─ vllm-xpu-kernels      ← native SYCL, **has GDN prefill** we need for     │
│  │                          Qwen3.5 linear-attn (chunk_gated_delta_rule_xe2) │
│  │                                                                           │
│  └─ custom-esimd-kernels-sglang  ← ESIMD-heavy decode paths (GDN *decode*,   │
│                                   fused QKV+norm+RoPE, fused norm+GEMV,      │
│                                   MoE batched ops). Not the same ops as     │
│                                   sgl-kernel-xpu; more aggressive fusion.    │
├──────────────────────────────────────────────────────────────────────────────┤
│  triton-xpu 3.7.0   (flaky on block-pointer-heavy kernels, e.g. GDN chunk)   │
│  torch.xpu          (upstream PyTorch XPU backend)                           │
│  IPEX / oneDNN / oneMKL / oneCCL / cutlass-sycl (FetchContent)               │
│  Level-Zero → compute-runtime → libze_intel_gpu → hardware                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Division of labor — practical summary:**

- `sgl-kernel-xpu` — the **default** implementation sglang pulls in on XPU; broad
  coverage, cutlass-sycl quality, cp39-SABI wheel, part of the stable sglang
  distribution.
- `vllm-xpu-kernels` — complements `sgl-kernel-xpu` with ops that **sglang
  doesn't yet have** (most importantly the **GDN linear-attn prefill** via
  `chunk_gated_delta_rule_xe2`, plus cutlass-based grouped GEMM and FA variants).
  We need it to lift the Qwen3.5 Triton-GDN fallback.
- `custom-esimd-kernels-sglang` — point solutions hand-tuned in ESIMD; valuable
  for **decode**-side hot paths (GDN decode, fused RMSNormGated+out_proj,
  fused QKV-split+norm+RoPE). Not prefill-capable for GDN, and shape-specialized.

## Build targets summary

| Target / source          | Compile                    | Used for                                                           |
|--------------------------|----------------------------|--------------------------------------------------------------------|
| `*.cpp` (non-Xe20)       | JIT / SPIR-V               | elementwise, RMSNorm, RoPE, quant, MoE helpers, sampling, spec     |
| `*Xe20.cpp` / `*Xe20.sycl` | **AOT `-device bmg`**    | cutlass-sycl FMHA decode / prefill / chunkprefill, MLA decode, Grouped-GEMM |
| `*.cpp.in`               | CMake-generated per shape  | flash-attn decode / prefill kernel instances, MLA decode, Group GEMM launcher |

## Scope & gaps relative to Qwen3.5 on XPU

- ✅ QKV RMSNorm + RoPE fusion (`fused_qk_norm_rope`) — matches Qwen3.5 attn.
- ✅ MLA decode (DeepSeek V2/V3), grouped GEMM for MoE, FP8/INT4/INT8 quant, sampling, spec decoding.
- ❌ **No GDN / linear-attention kernels** at all (neither prefill nor decode) — that's why we had to fall back to pure PyTorch for the extend path and ended up investigating `vllm-xpu-kernels` and `custom-esimd-kernels-sglang`.
- ❌ **No causal_conv1d** (Mamba-style) kernels — Qwen3.5 needs it at the start of every linear-attn layer.
- ❌ **No `ApplyShuffleMulSum` / `ep_moe_*` wiring yet into sglang main** for Qwen3.5 MoE path (those ops exist but aren't yet consumed).

The three packages are thus **complementary**, not alternatives.
