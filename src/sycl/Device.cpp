#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

static inline syclex::architecture get_device_architecture(at::DeviceIndex device_index = -1) {
  auto device_id = (device_index == -1) ? c10::xpu::current_device() : device_index;
  auto raw_device = c10::xpu::get_raw_device(device_id);
  return raw_device.get_info<syclex::info::device::architecture>();
}

std::tuple<int64_t, int64_t> query_device(int64_t device_index = -1) {
  auto device_arch = get_device_architecture(device_index);
  switch (device_arch) {
    case syclex::architecture::intel_gpu_bmg_g21:
    case syclex::architecture::intel_gpu_bmg_g31:
      return std::make_tuple(2, 0);
    // Xe3/Xe3P — Panther Lake (ptl_h, ptl_u) and Wildcat Lake (wcl).
    // As a workaround we reuse the Xe2 cutlass kernels on these chips
    // (mirrors vllm-xpu-kernels@368f685). The arch code is still reported
    // distinctly so Python-side heuristics can diverge later.
    case syclex::architecture::intel_gpu_ptl_h:
    case syclex::architecture::intel_gpu_ptl_u:
    case syclex::architecture::intel_gpu_wcl:
      return std::make_tuple(3, 0);
    // more arch is coming soon
    default:
      throw std::runtime_error("Unsupported XPU architecture.");
  }
}
