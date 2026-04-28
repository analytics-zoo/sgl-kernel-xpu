# sgl-kernel-xpu fork — integration lessons learned

Running log of what worked and what didn't while bringing up `intel/esimd-gdn-integration`. Keep this short; the details live in `BUILD_NOTES.md` and upstream `KERNELS.md`.

## The big regression we chased

### Symptom

`run_qwen35.py` (Qwen3.5-0.8B via `sglang.Engine`) crashes during sampling with:

```
/pytorch/third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/TensorCompareKernels.cpp:180:
operator(): Assertion `input_[0] != 0` failed.
Subprocess scheduler_0 (pid=N) crashed with exit code -6
```

Only surfaces **after** the scheduler starts producing tokens — prefill kernels all run, a few decode steps happen, then the assertion fires inside `process_batch_result_decode` when sglang calls `next_token_ids.tolist()` to sync.

Consequence: the next time anything tries to `torch.xpu.empty_cache()` it hangs in `urEventWait → sched_yield` forever (Level-Zero queue left with orphaned events). Only way out is `docker rm && docker run` a new container.

### The A/B matrix that found it

Fresh containers, editable sglang with XPU patches, same Qwen3.5-0.8B, same prompts:

| # | sgl_kernel                                        | Qwen3.5 |
|---|---------------------------------------------------|---------|
| 1 | **Stock** (image layer; `git+sgl-project/sgl-kernel-xpu` at image-build time) | ✅ |
| 2 | Fork pristine: cutlass `2fc09973`, GdnAttn OFF    | ✅ |
| 3 | Fork: cutlass `3f2a337e`, GdnAttn ON              | ❌ |
| 4 | Fork: cutlass `3f2a337e`, GdnAttn OFF             | ❌ |
| 5 | Fork: cutlass `2fc09973`, **GdnAttn ON**          | ✅ |

### Verdict

**cutlass-sycl revision bump `2fc09973 → 3f2a337e` is the regression.** Bumping cutlass alone (row 4) breaks Qwen3.5 even with the vendored gdn_attn sources entirely absent; pinning cutlass at `2fc09973` and adding the gdn_attn sources (row 5) works. The gdn_attn code is innocent.

The earlier "needs 3f2a337e for `barrier_arrive` / `cutlass::get_sub_group_id` / `get_block_2d_copy_D`" diagnosis was **wrong** — all three APIs already exist in `2fc09973`. The actual blocker on the first attempt was a missing `-DSYCL_INTEL_TARGET` compile definition, not a missing API.

### Why cutlass bump matters even if gdn_attn is unused

`cutlass-sycl` is **header-only**. It's pulled into upstream FMHA / MLA / Grouped-GEMM translation units regardless of whether our gdn_attn `.so` is loaded. Changing the cutlass revision changes the templates those TUs instantiate, which in turn affects the SPIR-V / GEN ISA of dozens of upstream kernels. One of those (we haven't pinned which) produces output that trips a `torch.where` / `torch.eq` assertion in `TensorCompareKernels.cpp` downstream.

This is a real upstream bug — cutlass-sycl `3f2a337e` has a codegen interaction with oneAPI 2025.3.3 / PyTorch 2.11.0+xpu / Battlemage G21 that silently corrupts some reduction or comparison output in a way that only manifests during sampler sync. Same test passes on stock / pinned revision.

## Working build combo (as of 2026-04-28)

```
sgl-kernel-xpu @ intel/esimd-gdn-integration
├── cutlass-sycl pin: 2fc09973  (upstream — DO NOT BUMP)
├── src/GdnAttn.cmake       ENABLED
├── src/Esimd.cmake         disabled (unrelated — see below)
└── src/sycl/Device.cpp     Xe3 WA applied (Panther/Wildcat Lake → Xe2)
```

Python side:
- `torch.ops.sgl_kernel.gdn_attention` present and callable
- upstream ops (rmsnorm / GEMV / FMHA / MLA / sampling / etc.) all preserved
- `query_device()` knows Xe3 and reports `(3, 0)`
- `is_xe2_or_xe3_arch()` gate present so MoE kernels will light up on Panther Lake too

sglang editable install with XPU patches (`chunk_torch_xpu.py` fallback, `_get_sm_count` XPU branch, `gdn_triton.py::is_xpu()` fallback dispatch).

## ESIMD is still disabled

Orthogonal problem. `src/Esimd.cmake` can't go through `sycl_add_library()` without hitting two DPC++ conflicts:

1. `sycl_add_library()` always emits `-fsycl-host-compiler=gcc` via `run_sycl.cmake:75`. That's incompatible with `-fsycl-unnamed-lambda`, which the vendored `qkv_split_norm_rope.h:252` relies on.
2. Without that flag, ESIMD `fp8_GEMM_pert.h` / `gemm.cpp` pull in `std/experimental/simd.hpp` through a header path that then fights with `sycl/ext/intel/esimd/detail/types_elementary.hpp` and fails template-deduction.

Fixing either would unlock ESIMD decode-path ops (rms_norm_gated, gdn_conv_fused_seq, moe batch, eagle). Not blocking Qwen3.5 prefill work, so parked for now.

## Runtime caveats discovered along the way

### Container state poisoning

A crashed sglang run (anything that kills the scheduler process mid-kernel) leaves Level-Zero events on the device. `docker restart` does **not** clear them — only `docker rm && docker run` does. Every Qwen3.5 experiment after a crash must be preceded by container recreation, otherwise subsequent `torch.xpu.empty_cache()` hangs forever.

### Don't probe without a hard timeout

`timeout --signal=KILL 180 python3 run_qwen35.py` around every test. Without it, a hung scheduler sits at 99% CPU for hours and our tool's `timeout` parameter only terminates the Bash wrapper, not the detached sglang scheduler subprocesses.

### `sgl_kernel` uninstall resets editable sglang

When we `pip uninstall sgl-kernel && pip install ...` into a fresh container, sglang editable install is fine only if the container's `/opt/venv/lib/python3.12/site-packages/sglang.egg-link` is still pointing at `/llm/workspace/sglang/python`. Container recreation loses this; reinstall with `pip install --no-build-isolation --no-deps -e .` each time.

### Proxy

Every fresh container needs `http_proxy=http://proxy.iil.intel.com:911` exported before any `pip install` that hits PyPI (including the scikit-build-core bootstrap).

## Qwen3.5 kernel wiring status (as of 2026-04-28)

After the cutlass `2fc09973` pin, audit of `qwen3_5.py` → `layers/` shows the
following op → XPU-backing mapping. "Already native" means sglang's
`MultiPlatformOp` subclass has a `forward_xpu` that dispatches to
`torch.ops.sgl_kernel.*`, which is registered for `kXPU` in
`src/torch_extension_sycl.cc` and exists in the current fork build.

| Layer / op | sglang class | Dispatches to | Status |
|---|---|---|---|
| GemmaRMSNorm (input/post/q/k norms) | `layernorm.py::GemmaRMSNorm.forward_xpu` | `sgl_kernel.gemma_rmsnorm` / `gemma_fused_add_rmsnorm` | ✅ native |
| Full-attn RoPE | `rotary_embedding/base.py::RotaryEmbedding.forward_xpu` | `sgl_kernel.rotary_embedding` | ✅ native |
| Full-attn MHA (RadixAttention) | `layers/attention/xpu_backend.py` | sgl-kernel-xpu cutlass FMHA | ✅ native |
| MLP `silu_and_mul` | `activation.py::SiluAndMul.forward_xpu` | `sgl_kernel.silu_and_mul` | ✅ native |
| GDN conv1d | `fla/causal_conv1d_triton.py` | Triton on XPU | ⚠️ Triton (works) |
| GDN `fused_qkvzba_split_reshape_cat_contiguous` | Triton | Triton on XPU | ⚠️ Triton (works) |
| GDN `fused_gdn_gating` | Triton | Triton on XPU | ⚠️ Triton (works) |
| GDN `chunk_gated_delta_rule` (extend) | `fla/chunk*.py` → XPU branch | **PyTorch eager (`chunk_torch_xpu.py`)** | ❌ slow fallback |
| GDN decode | `fused_sigmoid_gating_delta_rule_update` Triton | Triton on XPU | ⚠️ Triton (works) |
| RMSNormGated | `fla/layernorm_gated.py` | Triton (with `_get_sm_count` XPU patch) | ⚠️ Triton (works) |
| `fused_qk_norm_rope` for full-attn | n/a | **Not applicable** — Qwen3.5's `attn_output_gate=True` makes qkv_proj emit `[q, gate, k, v]` instead of `[q, k, v]`. Fused kernel expects contiguous `[q|k|v]`; pre-packing copy would erase the fusion win on a 6-layer / 3072-wide QKV. Keep non-fused path (already native per row 1–2). | 🟰 keep split |

**Biggest remaining runtime loss:** GDN extend's PyTorch-eager fallback on 18
of 24 layers. Wiring `torch.ops.sgl_kernel.gdn_attention` (vendored, already
callable) requires resolving the MambaPool conv-state layout mismatch below.

## Outstanding action items

- [ ] File upstream issue against `intel/sycl-tla`: cutlass-sycl `3f2a337e` under oneAPI 2025.3.3 causes silent correctness regression in sgl-kernel-xpu FMHA/MLA (Qwen3.5 sampling path). Reproducer: swap GIT_TAG in sgl-kernel-xpu's CMakeLists.txt and rerun `run_qwen35.py`.
- [x] **Radix cache on XPU — unblocked via `extra_buffer`, verified 2026-04-28.** sglang `server_args.py:2343` had `assert is_cuda()` on the `mamba extra_buffer` path; widened to `is_cuda() or is_xpu()`. sglang `mem_cache/memory_pool.py:296,311` also hardcoded `device="cuda"` for `intermediate_ssm_state_cache` and `intermediate_conv_window_cache` — switched to the function's `device` parameter. To use: pass `mamba_scheduler_strategy="extra_buffer"` (default is still `no_buffer`) and leave `disable_radix_cache=False`. XPU `page_size` stays 128 (the existing attention-backend constraint); `FLA_CHUNK_SIZE=64` and default `mamba_track_interval=256` both divide/are divided by 128 so the surrounding divisibility asserts pass. **Verified end-to-end on Arc Pro B60 / oneAPI 2025.3.3 / PyTorch 2.11.0+xpu:** a Qwen3.5-0.8B run (3 prompts × 96 tokens, GPU 0 via `ONEAPI_DEVICE_SELECTOR=level_zero:0`) completes with `rc=0` and clean engine shutdown; ServerArgs at startup confirms `mamba_scheduler_strategy='extra_buffer'` / `disable_radix_cache=False`.
- [x] **MambaPool `conv_state` layout adapter — option 1 implemented** (gather/scatter around the kernel call, in `Qwen3_5GatedDeltaNet._forward_xpu_fast_path`). Adapter also gathers/scatters `ssm_state`, and passes an identity `scratch_indices` to the kernel since both state tensors are now densely packed. Gated by env var `SGLANG_XPU_GDN_FAST_PATH=1` so the default behavior (PyTorch fallback) is preserved until the path is validated end-to-end. To A/B: `SGLANG_XPU_GDN_FAST_PATH=1 timeout --signal=KILL 180 python3 run_qwen35.py`.
- [ ] Validate correctness of the fast path against the PyTorch fallback (run `run_qwen35.py` with and without the env var, compare first-N-token outputs). Current gate: OFF by default.
- [ ] ESIMD bring-up (separate branch): either rename all ESIMD unnamed lambdas to named kernels, or write an `esimd_add_library` helper that bypasses `sycl_add_library`'s `-fsycl-host-compiler` injection.
- [ ] Validate Qwen3.5-35B-A3B MoE path once ESIMD MoE ops land.

## Useful invariants

- Pristine fork rebuild takes 30–40 min (see `BUILD_NOTES.md`). Don't `rm -rf build/` unless you're changing CMake structure; swap `GIT_TAG` on cutlass and ninja will handle the rest.
- `0 errors` in build log does NOT mean "runtime correct" — it means "compile+link succeeded". cutlass `3f2a337e` compiled perfectly cleanly and produced a wheel; correctness came apart at the sampler call site.
- Qwen3.5 prefill uses 18 of 24 layers going through the GDN path; 6 go through full attention. So any FMHA regression will surface on every Qwen3.5 prompt, not just the linear-attn layers.
