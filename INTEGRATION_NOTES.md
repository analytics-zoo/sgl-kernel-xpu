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

## Triton kernels on the Qwen3.5 XPU forward path

Exhaustive list (per-forward-call) of Triton kernels we currently execute on
Intel XPU during a `run_qwen35.py`-style run. Dropped kernels that the
pipeline doesn't actually hit (NSA, MLA, FP4, AIter, AWQ Triton, MoE router,
TRTLLM, Flash backends, `fused_softcap` — Qwen3.5-0.8B `final_logit_softcapping=None`).

Status legend: ✅ runs correctly / ⚠️ runs but a SYCL replacement would be
faster / ❌ doesn't compile on Triton-XPU.

| # | Triton kernel | File | Called from | Per-token cost | XPU status | Replacement plan |
|---|---|---|---|---|---|---|
| 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | `jit_kernel/triton/gdn_fused_proj.py:159` | `Qwen3_5GatedDeltaNet.forward` (1× per GDN layer) | 18 layers × 1 | ✅ | **subsumed** by `torch.ops.sgl_kernel.gdn_attention` — the fast path takes raw `projected_states_qkvz` / `_ba` and does the split inside the SYCL kernel. No separate replacement needed once the fast path is on. |
| 2 | `_causal_conv1d_fwd_kernel` (extend) | `layers/attention/mamba/causal_conv1d_triton.py:14` | `GDNAttnBackend.forward_extend` | 18 layers × 1 (prefill only) | ✅ | **subsumed** by `gdn_attention` (kernel internally runs `chunk_causal_conv1d_xe2`). |
| 3 | `_causal_conv1d_update_kernel` (decode) | `layers/attention/mamba/causal_conv1d_triton.py:570` | `GDNAttnBackend.forward_decode` | 18 layers × 1 (decode only) | ✅ | **subsumed** by `gdn_attention` (its decode branch uses the same conv path). |
| 4 | `fused_gdn_gating_kernel` | `layers/attention/fla/fused_gdn_gating.py:10` | `GDNAttnBackend.forward_extend` + `_decode` | 18 layers × 1 | ✅ | **subsumed** by `gdn_attention` (gating is fused into the SYCL chunk kernel). |
| 5 | `chunk_*_kernel` cluster (8 files: `chunk_delta_h.py`, `chunk_fwd.py`, `chunk_intra*.py`, `chunk_o.py`, `chunk_scaled_dot_kkt.py`, `solve_tril.py`, `wy_fast.py`) | `layers/attention/fla/chunk*.py` | `chunk_gated_delta_rule` (extend path) | 18 layers × chunks | ❌ **does not compile** — Triton-XPU 3.7.0 `TritonIntelStrideVersioning` asserts on the block-pointer pattern | **already patched**: XPU branch in `gdn_triton.py` routes to `chunk_gated_delta_rule_torch` (pure PyTorch). Will be **subsumed** by `gdn_attention` once fast path validates. |
| 6 | `fused_sigmoid_gating_delta_rule_update_kernel` | `layers/attention/fla/fused_sigmoid_gating_recurrent.py:8` | `TritonGDNKernel.decode` / `target_verify` | 18 layers × 1 (decode only) | ✅ | **subsumed** by `gdn_attention` decode branch. Still Triton-only today when fast path is off. |
| 7 | `_layer_norm_fwd_1pass_kernel` (RMSNormGated) | `layers/attention/fla/layernorm_gated.py:67` | `Qwen3_5GatedDeltaNet.forward` (post-attn norm gate) | 18 layers × 1 | ✅ (needs `_get_sm_count` XPU patch — already landed) | **partially subsumed** — the `gdn_attention` fast path only covers conv1d+GDN+RMSNormGated fusion if we feed its z-output through the same call; currently we still call `self.norm(core_attn_out, z)` after the kernel. Option: either leave Triton (works) or port to `sgl_kernel.gemma_rmsnorm` + manual SiLU(z) multiply — only worth it if profiling shows this norm is >5% of per-layer time. |
| 8 | `fused_softcap_kernel` | `layers/logits_processor.py:1109` | Only if `final_logit_softcapping` is set | **not triggered** for Qwen3.5-0.8B | n/a | Skip. |

### Net picture

Every Qwen3.5 Triton kernel except `_layer_norm_fwd_1pass_kernel` (row 7) is
subsumed by the single `torch.ops.sgl_kernel.gdn_attention` op once the fast
path is on. That means:

- **Before fast path validates**: rows 1–4, 6 run through Triton (all work),
  row 5 through PyTorch fallback (slow), row 7 through Triton (works).
- **After fast path validates** (`SGLANG_XPU_GDN_FAST_PATH=1`): only row 7
  still runs Triton. Eliminating that one is a small incremental win — much
  less than the row-5 savings.

So the priority order for Triton→SYCL replacement work is:

1. **Validate the `gdn_attention` fast path** → erases rows 1–6 in one go.
   Blocked only by A/B'ing output against `chunk_torch_xpu.py`.
2. **Profile row 7**. If `_layer_norm_fwd_1pass_kernel` is a measurable
   fraction of per-layer time, write a small SYCL kernel for it (the math is
   trivial: `rmsnorm(x) * weight * silu(z)` per head). ESIMD
   `rms_norm_gated.h` already implements this and is sitting in
   `src/esimd/esimd_kernels/` — the blocker is the ESIMD build (see action
   items). Alternative: compose `sgl_kernel.rmsnorm` + elementwise SiLU × z
   in Python; two ops + some dispatch overhead, but avoids the ESIMD
   toolchain fight.
3. Leave logits `fused_softcap` alone (dead code for this model).

### Other Triton files in the tree that are NOT hit by Qwen3.5

Listed so future auditors don't chase ghosts: `fla/cumsum.py`,
`fla/fused_norm_gate.py` (used by KDA), `fla/fused_recurrent.py`,
`fla/index.py`, `fla/kda.py`, `fla/l2norm.py`, `fla/op.py`, `fla/utils.py`,
`linear/lightning_attn.py`, `linear/seg_la.py`, `mamba/ops/ssd_*` (Mamba2,
not GDN), `mamba/mamba_state_scatter_triton.py`, plus the 60+ MoE / quant /
attention-backend files (`triton_backend.py`, `triton_ops/extend_attention.py`,
etc.) — none fire on the `intel_xpu` attention backend for a dense Qwen3.5.

## `cu_seqlens_knew` uninitialized pointer bug in `flash_attention.cpp`

Host-side bug in `src/sycl/flash_attention.cpp`. Both `decode::mha_fwd` and
`prefill::mha_fwd` populate `params.cu_seqlens_q` and `params.cu_seqlens_k`,
but **never initialize `params.cu_seqlens_knew`**. The prefill/decode
runners (`xe_fmha_fwd_{prefill,decode}_runner.hpp`) read
`params.cu_seqlens_knew` when `isVarLen=true` (which is the default path —
see `FMHAConfig::run(params)` fallback at `xe_fmha_fwd_prefill_runner.hpp:437`,
which calls `run<true, true, true, ...>(params)` —> `isVarLen=true`):

```cpp
shape.seq_len_qo.cumulative_length       = params.cu_seqlens_q;
shape.seq_len_kv.cumulative_length       = params.cu_seqlens_knew;  // uninit!
shape.seq_len_kv_cache.cumulative_length = params.cu_seqlens_k;
```

On host, `Arguments params;` is stack-allocated, so `cu_seqlens_knew` holds a
garbage pointer. The device code dereferences it to build the ragged-tensor
shape for "new K" tokens; on Qwen3.5 hybrid model's first full-attention
layer (L3, head_dim=256, num_kv_heads=2) under sglang server load, this
reliably trips `torch-xpu-ops/src/ATen/native/xpu/sycl/Indexing.h:622
Assertion index out of bounds failed`. Qwen3-0.6B (head_dim=128) didn't
trip it — probably just lucky that the garbage happened to point to a
readable-enough region in its param layout.

**Symptom**: single greedy prefill request to sglang server hangs; py-spy
shows `process_batch_result_decode:413` → `next_token_ids.tolist()` →
`urEventWait → sched_yield` forever, and kern.log shows
`xe 0000:18:00.0 [drm] Tile0: GT0: Engine reset: engine_class=ccs` at the
time of the stuck forward.

**Fix**: alias `cu_seqlens_knew` to `cu_seqlens_k` at the host side in both
mha_fwd functions — for sglang's call pattern all K tokens are "new" from
the kernel's perspective, so the two cumulative-length arrays are
identical. (If sglang ever passes a separate `cu_seqlens_k_new`, they can
be wired independently.) See `src/sycl/flash_attention.cpp` ~L310, L598.

**Residual**: after the fix, the Indexing.h:622 assertion no longer fires
and GPU keeps running, but sglang's `server + Qwen3.5 + chunked prefill`
combination still triggers a GPU `engine_class=ccs` reset at the boundary
between the last prefill chunk and the first decode step — possibly a
second submission-pattern bug, not yet root-caused.

## Outstanding action items

- [ ] File upstream issue against `intel/sycl-tla`: cutlass-sycl `3f2a337e` under oneAPI 2025.3.3 causes silent correctness regression in sgl-kernel-xpu FMHA/MLA (Qwen3.5 sampling path). Reproducer: swap GIT_TAG in sgl-kernel-xpu's CMakeLists.txt and rerun `run_qwen35.py`.
- [x] **`cu_seqlens_knew` uninitialized pointer fix landed** in `flash_attention.cpp` — eliminates the Indexing.h:622 assertion on Qwen3.5 full-attention path under sglang server. See dedicated section above.
- [ ] **Residual `engine_class=ccs` reset on Qwen3.5 server decode** — after the cu_seqlens_knew fix the Indexing.h assertion is gone, but server requests still don't complete because GPU engine resets at the last-prefill → first-decode boundary. Offline (`run_qwen35_radix.py`) still works. Not root-caused.
- [x] **Radix cache on XPU — unblocked via `extra_buffer`, verified 2026-04-28.** sglang `server_args.py:2343` had `assert is_cuda()` on the `mamba extra_buffer` path; widened to `is_cuda() or is_xpu()`. sglang `mem_cache/memory_pool.py:296,311` also hardcoded `device="cuda"` for `intermediate_ssm_state_cache` and `intermediate_conv_window_cache` — switched to the function's `device` parameter. To use: pass `mamba_scheduler_strategy="extra_buffer"` (default is still `no_buffer`) and leave `disable_radix_cache=False`. XPU `page_size` stays 128 (the existing attention-backend constraint); `FLA_CHUNK_SIZE=64` and default `mamba_track_interval=256` both divide/are divided by 128 so the surrounding divisibility asserts pass. **Verified end-to-end on Arc Pro B60 / oneAPI 2025.3.3 / PyTorch 2.11.0+xpu:** a Qwen3.5-0.8B run (3 prompts × 96 tokens, GPU 0 via `ONEAPI_DEVICE_SELECTOR=level_zero:0`) completes with `rc=0` and clean engine shutdown; ServerArgs at startup confirms `mamba_scheduler_strategy='extra_buffer'` / `disable_radix_cache=False`.
- [x] **MambaPool `conv_state` layout adapter — option 1 implemented** (gather/scatter around the kernel call, in `Qwen3_5GatedDeltaNet._forward_xpu_fast_path`). Adapter also gathers/scatters `ssm_state`, and passes an identity `scratch_indices` to the kernel since both state tensors are now densely packed. Gated by env var `SGLANG_XPU_GDN_FAST_PATH=1` so the default behavior (PyTorch fallback) is preserved until the path is validated end-to-end. To A/B: `SGLANG_XPU_GDN_FAST_PATH=1 timeout --signal=KILL 180 python3 run_qwen35.py`.
- [ ] Validate correctness of the fast path against the PyTorch fallback (run `run_qwen35.py` with and without the env var, compare first-N-token outputs). Current gate: OFF by default.
- [ ] ESIMD bring-up (separate branch): either rename all ESIMD unnamed lambdas to named kernels, or write an `esimd_add_library` helper that bypasses `sycl_add_library`'s `-fsycl-host-compiler` injection.
- [ ] Validate Qwen3.5-35B-A3B MoE path once ESIMD MoE ops land.

## Useful invariants

- Pristine fork rebuild takes 30–40 min (see `BUILD_NOTES.md`). Don't `rm -rf build/` unless you're changing CMake structure; swap `GIT_TAG` on cutlass and ninja will handle the rest.
- `0 errors` in build log does NOT mean "runtime correct" — it means "compile+link succeeded". cutlass `3f2a337e` compiled perfectly cleanly and produced a wheel; correctness came apart at the sampler call site.
- Qwen3.5 prefill uses 18 of 24 layers going through the GDN path; 6 go through full attention. So any FMHA regression will surface on every Qwen3.5 prompt, not just the linear-attn layers.
