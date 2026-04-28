# Copyright 2025 SGLang Team. All Rights Reserved.
# Thin re-export of the ESIMD ops vendored from custom-esimd-kernels-sglang.
# All ops are registered under `torch.ops.sgl_kernel.*` (see
# src/sycl/esimd/torch_bindings_esimd.cc). Detailed docstrings on argument
# shapes/dtypes live with the corresponding implementations in
# src/sycl/esimd/esimd_kernels/.

import torch

_ops = torch.ops.sgl_kernel

# ---- FP8 / INT4 GEMV ------------------------------------------------------
esimd_gemv_fp8_pern         = _ops.esimd_gemv_fp8_pern
esimd_gemv_fp8_pern_fused2  = _ops.esimd_gemv_fp8_pern_fused2
esimd_gemv_fp8_pern_fused3  = _ops.esimd_gemv_fp8_pern_fused3
esimd_gemv_fp8_pert         = _ops.esimd_gemv_fp8_pert
esimd_gemv_fp8_pert_fused2  = _ops.esimd_gemv_fp8_pert_fused2
esimd_gemv_fp8_pert_fused3  = _ops.esimd_gemv_fp8_pert_fused3
esimd_gemv_int4             = _ops.esimd_gemv_int4
esimd_gemv_int4_fused2      = _ops.esimd_gemv_int4_fused2

# ---- QKV split + norm + RoPE ---------------------------------------------
esimd_qkv_split_norm_rope   = _ops.esimd_qkv_split_norm_rope

# ---- Fused RMSNorm variants ----------------------------------------------
esimd_fused_add_rms_norm         = _ops.esimd_fused_add_rms_norm
esimd_rms_norm_gated             = _ops.esimd_rms_norm_gated
esimd_fused_add_rms_norm_batched = _ops.esimd_fused_add_rms_norm_batched

# ---- Fused ResAdd + RMSNorm + GEMV ---------------------------------------
esimd_resadd_norm_gemv_fp8_pert  = _ops.esimd_resadd_norm_gemv_fp8_pert
esimd_resadd_norm_gemv2_fp8_pert = _ops.esimd_resadd_norm_gemv2_fp8_pert
esimd_resadd_norm_gemv_int4_pert = _ops.esimd_resadd_norm_gemv_int4_pert

# ---- Fused RMSNormGated + GEMV (GDN out_proj) ----------------------------
esimd_norm_gemv_fp8_pert    = _ops.esimd_norm_gemv_fp8_pert
esimd_norm_gemv_int4_pert   = _ops.esimd_norm_gemv_int4_pert

# ---- GDN Conv1d-fused decode (doubleGRF, AOT BMG) ------------------------
esimd_gdn_conv_fused        = _ops.esimd_gdn_conv_fused
esimd_gdn_conv_fused_seq    = _ops.esimd_gdn_conv_fused_seq

# ---- MoE auxiliary (scatter/gather/topk/silu_mul) ------------------------
esimd_moe_topk              = _ops.esimd_moe_topk
esimd_moe_scatter           = _ops.esimd_moe_scatter
esimd_moe_scatter_fused     = _ops.esimd_moe_scatter_fused
esimd_moe_silu_mul          = _ops.esimd_moe_silu_mul
esimd_moe_gather            = _ops.esimd_moe_gather
esimd_moe_gemm_fp8          = _ops.esimd_moe_gemm_fp8
esimd_moe_gemm_fp8_pert     = _ops.esimd_moe_gemm_fp8_pert
esimd_moe_topk_v2           = _ops.esimd_moe_topk_v2

# ---- FP8 GEMM M>=1 --------------------------------------------------------
esimd_gemm_fp8_pert         = _ops.esimd_gemm_fp8_pert
