/* Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Merged ESIMD op registrations for sgl-kernel-xpu (Intel Arc Pro B60 / BMG).
 * Ports the four upstream .cc files from custom-esimd-kernels-sglang/csrc/xpu/:
 *   - torch_extension.cc       (core: gemv_fp8, norm+gemv, qkv_split_norm_rope, rms_norm_gated)
 *   - torch_extension_lgrf.cc  (doubleGRF GDN conv-fused decode)
 *   - torch_extension_moe.cc   (MoE aux ops: topk, scatter, silu_mul, gather, gemm_fp8)
 *   - torch_extension_gemm.cc  (FP8 GEMM M>=1)
 *
 * All schemas are placed under the `sgl_kernel` torch-library namespace so they
 * are reachable via `torch.ops.sgl_kernel.<op>` alongside the native SYCL ops.
 * (torch_extension_topk_v2.cc is merged too; it registered its own
 * `esimd_topk_v2` namespace upstream — we collapse it into `sgl_kernel` here.)
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "esimd_kernel_ops.h"

// Forward declaration for the TopK V2 entry (formerly in its own .cc).
at::Tensor esimd_moe_topk_v2(
    at::Tensor router_logits, at::Tensor top_values, at::Tensor top_indices,
    int64_t T, int64_t num_experts, int64_t topk);

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  // -------- core ESIMD ops (formerly custom_esimd_kernels) --------
  m.def("esimd_gemv_fp8_pern(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output, int N, int K) -> Tensor");
  m.impl("esimd_gemv_fp8_pern", torch::kXPU, &esimd_gemv_fp8_pern);

  m.def("esimd_gemv_fp8_pern_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, int N0, "
        "Tensor w1, Tensor s1, Tensor o1, int N1, "
        "int K) -> Tensor");
  m.impl("esimd_gemv_fp8_pern_fused2", torch::kXPU, &esimd_gemv_fp8_pern_fused2);

  m.def("esimd_gemv_fp8_pern_fused3(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, int N0, "
        "Tensor w1, Tensor s1, Tensor o1, int N1, "
        "Tensor w2, Tensor s2, Tensor o2, int N2, "
        "int K) -> Tensor");
  m.impl("esimd_gemv_fp8_pern_fused3", torch::kXPU, &esimd_gemv_fp8_pern_fused3);

  m.def("esimd_gemv_fp8_pert(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemv_fp8_pert", torch::kXPU, &esimd_gemv_fp8_pert);

  m.def("esimd_gemv_fp8_pert_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, "
        "Tensor w1, Tensor s1, Tensor o1) -> Tensor");
  m.impl("esimd_gemv_fp8_pert_fused2", torch::kXPU, &esimd_gemv_fp8_pert_fused2);

  m.def("esimd_gemv_fp8_pert_fused3(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, "
        "Tensor w1, Tensor s1, Tensor o1, "
        "Tensor w2, Tensor s2, Tensor o2) -> Tensor");
  m.impl("esimd_gemv_fp8_pert_fused3", torch::kXPU, &esimd_gemv_fp8_pert_fused3);

  m.def("esimd_gemv_int4(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemv_int4", torch::kXPU, &esimd_gemv_int4);

  m.def("esimd_gemv_int4_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, "
        "Tensor w1, Tensor s1, Tensor o1) -> Tensor");
  m.impl("esimd_gemv_int4_fused2", torch::kXPU, &esimd_gemv_int4_fused2);

  m.def("esimd_qkv_split_norm_rope(Tensor qkv_state, "
        "Tensor q_out, Tensor gate_out, Tensor k_out, Tensor v_out, "
        "Tensor norm_wq, Tensor norm_wk, Tensor positions, "
        "int q_heads, int kv_heads, bool attn_output_gate, "
        "int rotary_dim, Tensor cos_sin_cache) -> Tensor");
  m.impl("esimd_qkv_split_norm_rope", torch::kXPU, &esimd_qkv_split_norm_rope);

  m.def("esimd_resadd_norm_gemv_fp8_pert(Tensor hidden_states, Tensor residual, "
        "Tensor norm_weight, Tensor gemv_weight, Tensor gemv_scale, "
        "Tensor output, Tensor normed_out, float eps) -> Tensor");
  m.impl("esimd_resadd_norm_gemv_fp8_pert", torch::kXPU, &esimd_resadd_norm_gemv_fp8_pert);

  m.def("esimd_resadd_norm_gemv2_fp8_pert(Tensor hidden_states, Tensor residual, "
        "Tensor norm_weight, "
        "Tensor w0, Tensor s0, Tensor o0, "
        "Tensor w1, Tensor s1, Tensor o1, "
        "float eps) -> Tensor");
  m.impl("esimd_resadd_norm_gemv2_fp8_pert", torch::kXPU, &esimd_resadd_norm_gemv2_fp8_pert);

  m.def("esimd_norm_gemv_fp8_pert(Tensor x, Tensor z, Tensor norm_weight, "
        "Tensor gemv_weight, Tensor gemv_scale, Tensor output, "
        "int HV, int V, float eps) -> Tensor");
  m.impl("esimd_norm_gemv_fp8_pert", torch::kXPU, &esimd_norm_gemv_fp8_pert);

  m.def("esimd_resadd_norm_gemv_int4_pert(Tensor hidden_states, Tensor residual, "
        "Tensor norm_weight, Tensor gemv_weight, Tensor gemv_scale, "
        "Tensor output, Tensor normed_out, float eps) -> Tensor");
  m.impl("esimd_resadd_norm_gemv_int4_pert", torch::kXPU, &esimd_resadd_norm_gemv_int4_pert);

  m.def("esimd_norm_gemv_int4_pert(Tensor x, Tensor z, Tensor norm_weight, "
        "Tensor gemv_weight, Tensor gemv_scale, Tensor output, "
        "int HV, int V, float eps) -> Tensor");
  m.impl("esimd_norm_gemv_int4_pert", torch::kXPU, &esimd_norm_gemv_int4_pert);

  m.def("esimd_fused_add_rms_norm(Tensor hidden_states, Tensor residual, "
        "Tensor weight, float eps) -> Tensor");
  m.impl("esimd_fused_add_rms_norm", torch::kXPU, &esimd_fused_add_rms_norm);

  m.def("esimd_rms_norm_gated(Tensor x, Tensor z, Tensor weight, "
        "Tensor output, float eps) -> Tensor");
  m.impl("esimd_rms_norm_gated", torch::kXPU, &esimd_rms_norm_gated);

  m.def("esimd_fused_add_rms_norm_batched(Tensor hidden_states, Tensor residual, "
        "Tensor weight, float eps) -> Tensor");
  m.impl("esimd_fused_add_rms_norm_batched", torch::kXPU, &esimd_fused_add_rms_norm_batched);

  // -------- doubleGRF ops (formerly custom_esimd_kernels_lgrf) --------
  m.def("esimd_gdn_conv_fused(Tensor qkvz, "
        "Tensor conv_state, Tensor conv_weight, Tensor conv_bias, "
        "Tensor conv_state_indices, "
        "Tensor A_log, Tensor dt_bias, "
        "Tensor ba, "
        "Tensor ssm_state, Tensor ssm_state_indices, "
        "Tensor output, Tensor z_out, "
        "int N, int H, int HV, int K, int V, "
        "float scale) -> Tensor");
  m.impl("esimd_gdn_conv_fused", torch::kXPU, &esimd_gdn_conv_fused);

  m.def("esimd_gdn_conv_fused_seq(Tensor qkvz, "
        "Tensor conv_state, Tensor conv_weight, Tensor conv_bias, "
        "Tensor conv_state_indices, "
        "Tensor A_log, Tensor dt_bias, "
        "Tensor ba, "
        "Tensor ssm_state, Tensor ssm_state_indices, "
        "Tensor output, Tensor z_out, "
        "int N, int H, int HV, int K, int V, "
        "float scale) -> Tensor");
  m.impl("esimd_gdn_conv_fused_seq", torch::kXPU, &esimd_gdn_conv_fused_seq);

  // -------- MoE auxiliary ops (formerly custom_esimd_kernels_moe) --------
  m.def("esimd_moe_topk(Tensor router_logits, Tensor top_values, "
        "Tensor top_indices, int T) -> Tensor");
  m.impl("esimd_moe_topk", torch::kXPU, &esimd_moe_topk);

  m.def("esimd_moe_scatter(Tensor hidden_states, Tensor router_top_value, "
        "Tensor sorted_token_ids, "
        "Tensor scattered_hidden, Tensor scattered_weights, "
        "int K, int topk, int total_expanded) -> Tensor");
  m.impl("esimd_moe_scatter", torch::kXPU, &esimd_moe_scatter);

  m.def("esimd_moe_scatter_fused(Tensor hidden_states, Tensor top_values, "
        "Tensor top_indices, "
        "Tensor scattered_hidden, Tensor scattered_weights, "
        "Tensor topk_ids, Tensor expert_start, Tensor max_tokens_out, "
        "int K, int topk, int T, int num_experts) -> Tensor");
  m.impl("esimd_moe_scatter_fused", torch::kXPU, &esimd_moe_scatter_fused);

  m.def("esimd_moe_silu_mul(Tensor input, Tensor output, "
        "int N_gate_up, int N_half, int total_rows) -> Tensor");
  m.impl("esimd_moe_silu_mul", torch::kXPU, &esimd_moe_silu_mul);

  m.def("esimd_moe_gather(Tensor moe_output, Tensor topk_ids, "
        "Tensor scattered_weights, Tensor final_hidden, "
        "int K, int topk, int T) -> Tensor");
  m.impl("esimd_moe_gather", torch::kXPU, &esimd_moe_gather);

  m.def("esimd_moe_gemm_fp8(Tensor input, Tensor weight, Tensor scale, "
        "Tensor output, Tensor expert_idx, "
        "int N, int K, int num_experts, int max_tokens_per_expert) -> Tensor");
  m.impl("esimd_moe_gemm_fp8", torch::kXPU, &esimd_moe_gemm_fp8);

  m.def("esimd_moe_gemm_fp8_pert(Tensor input, Tensor weight, Tensor scale, "
        "Tensor output, Tensor expert_idx, "
        "int N, int K, int num_experts, int max_tokens_per_expert) -> Tensor");
  m.impl("esimd_moe_gemm_fp8_pert", torch::kXPU, &esimd_moe_gemm_fp8_pert);

  // -------- FP8 GEMM M>=1 (formerly custom_esimd_kernels_gemm) --------
  m.def("esimd_gemm_fp8_pert(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemm_fp8_pert", torch::kXPU, &esimd_gemm_fp8_pert);

  // -------- TopK V2 (formerly esimd_topk_v2 namespace) --------
  m.def("esimd_moe_topk_v2(Tensor router_logits, Tensor top_values, "
        "Tensor top_indices, int T, int num_experts, int topk) -> Tensor");
  m.impl("esimd_moe_topk_v2", torch::kXPU, &esimd_moe_topk_v2);
}
