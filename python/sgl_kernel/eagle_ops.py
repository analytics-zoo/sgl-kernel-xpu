# Copyright 2025 SGLang Team. All Rights Reserved.
# Wrappers for the Eagle speculative-decode ops vendored from
# custom-esimd-kernels-sglang/csrc/eagle/eagle.sycl. Those kernels keep their
# own torch-library namespace (`eagle_ops`) because they register per-tensor
# mutability markers in the schema — a convention the upstream ESIMD author
# picked and which existing sglang callers already know.

import torch

_eagle_ops = torch.ops.eagle_ops

eagle_gdn              = _eagle_ops.gdn_eagle
eagle_page_attn_decode = _eagle_ops.page_attn_decode
