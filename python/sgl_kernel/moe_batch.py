# Copyright 2025 SGLang Team. All Rights Reserved.
# Wrappers for the batched MoE ops vendored from
# custom-esimd-kernels-sglang/csrc/moe_batch/. Registered under their original
# torch-library namespaces (`moe_ops`, `moe_int4_ops`) to preserve the
# upstream API surface.

import torch

_moe      = torch.ops.moe_ops
_moe_int4 = torch.ops.moe_int4_ops

# ---- FP8 MoE batch --------------------------------------------------------
moe_router_forward    = _moe.moe_router_forward
moe_batch_topk        = _moe.moe_topk
moe_up_forward        = _moe.moe_up_forward
moe_down_forward      = _moe.moe_down_forward
moe_accumulate        = _moe.moe_accumulate
moe_forward_fused     = _moe.moe_forward_fused
moe_forward_full      = _moe.moe_forward_full
moe_forward_full_v2   = _moe.moe_forward_full_v2

# ---- INT4 MoE batch -------------------------------------------------------
moe_router_forward_int4 = _moe_int4.moe_router_forward_int4
moe_forward_full_int4   = _moe_int4.moe_forward_full_int4
