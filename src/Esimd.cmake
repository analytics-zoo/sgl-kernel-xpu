# Esimd.cmake — vendored from custom-esimd-kernels-sglang. Uses ESIMD
# (Explicit SIMD) SYCL extension.
#
# These .cpp files are SYCL kernels (with ESIMD intrinsics), so they must be
# compiled through icpx, not g++. We reuse upstream's sycl_add_library()
# helper (same as GdnAttn.cmake) and route ESIMD-specific compile flags
# through it so the ESIMD headers and AOT/doubleGRF options work.
#
# Layout under src/esimd/:
#   esimd_kernel.cpp                   - core ESIMD ops (JIT)
#   esimd_kernel_gemm.cpp              - FP8 GEMM M>=1 (JIT, DPAS — no AOT)
#   esimd_kernel_moe.cpp               - MoE aux ops (JIT, no doubleGRF)
#   esimd_kernel_lgrfXe20.cpp          - GDN conv-fused (AOT BMG + doubleGRF)
#   esimd_kernel_topk_v2Xe20.cpp       - TopK V2 (AOT BMG)
#   eagle/eagle.cpp                    - Eagle spec-decode (JIT)
#   moe_batch/moe.cpp + moe_int4.cpp   - MoE batch FP8 / INT4 (JIT)
#   esimd_kernels/*.h                  - headers (included from above)
#   torch_bindings_esimd.cc            - TORCH_LIBRARY_FRAGMENT host code
#   esimd_kernel_ops.h                 - forward-decl header

set(ESIMD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/esimd)

file(GLOB ESIMD_HOST_CC "${ESIMD_DIR}/*.cc")

# JIT-compiled ESIMD sources: no AOT, no doubleGRF.
set(ESIMD_JIT_SRCS
    "${ESIMD_DIR}/esimd_kernel.cpp"
    "${ESIMD_DIR}/esimd_kernel_gemm.cpp"
    "${ESIMD_DIR}/esimd_kernel_moe.cpp"
    "${ESIMD_DIR}/eagle/eagle.cpp"
    "${ESIMD_DIR}/moe_batch/moe.cpp"
    "${ESIMD_DIR}/moe_batch/moe_int4.cpp")

# AOT-compiled ESIMD sources for BMG (no doubleGRF).
set(ESIMD_AOT_SRCS
    "${ESIMD_DIR}/esimd_kernel_topk_v2Xe20.cpp")

# AOT + doubleGRF (512 regs / thread). One file only.
set(ESIMD_LGRF_SRC
    "${ESIMD_DIR}/esimd_kernel_lgrfXe20.cpp")

# ---- Host torch bindings (plain C++): link into common_ops ----
target_sources(common_ops PRIVATE ${ESIMD_HOST_CC})
target_include_directories(common_ops PRIVATE
    ${ESIMD_DIR}
    ${ESIMD_DIR}/esimd_kernels)

# ---- Helpers to build each bucket via sycl_add_library ----
# Upstream sycl_add_library() invokes icpx; we pass ESIMD-specific offline
# compiler flags through the second positional arg.

# ESIMD JIT flag bundle: no AOT target → let icpx JIT at runtime.
# We still want per-kernel device code split so ESIMD intrinsics don't collide.
set(_ESIMD_JIT_OFFLINE_FLAGS)

# ESIMD AOT flag bundle for BMG (same device name sgl-kernel-xpu uses).
set(_ESIMD_AOT_OFFLINE_FLAGS "-device bmg${SYCL_OFFLINE_COMPILER_CG_OPTIONS}")

# ESIMD AOT + doubleGRF (lgrf) flag bundle.
set(_ESIMD_LGRF_OFFLINE_FLAGS "-device bmg -doubleGRF${SYCL_OFFLINE_COMPILER_CG_OPTIONS}")

set(_ESIMD_INCLUDE_DIRS
    ${ESIMD_DIR}
    ${ESIMD_DIR}/esimd_kernels
    ${ESIMD_DIR}/eagle
    ${ESIMD_DIR}/moe_batch
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${SYCL_INCLUDE_DIR})

function(_add_esimd_lib sycl_lib offline_flags sources)
  # sycl_add_library() / run_sycl.cmake bakes SYCL_COMPILE_FLAGS into the
  # generated command file at configure time. Upstream BuildFlags.cmake hard-
  # codes -fno-sycl-unnamed-lambda, but ESIMD vendored headers use unnamed
  # parallel_for lambdas and need it enabled. We temporarily rewrite the
  # global var for the duration of this call, and also add the ESIMD-specific
  # -ffast-math / -fsycl-device-code-split=per_kernel.
  set(_saved_flags ${SYCL_COMPILE_FLAGS})
  # Remove -fno-sycl-unnamed-lambda: ESIMD vendored headers use unnamed
  # parallel_for lambdas. We can't just add -fsycl-unnamed-lambda because it
  # conflicts with -fsycl-host-compiler (which run_sycl.cmake always adds) —
  # so instead rely on the dpc++ 2025.3 default (= unnamed lambdas allowed
  # when the flag is absent).
  list(REMOVE_ITEM SYCL_COMPILE_FLAGS -fno-sycl-unnamed-lambda)
  list(APPEND SYCL_COMPILE_FLAGS
      -ffast-math
      -fsycl-device-code-split=per_kernel)

  sycl_add_library(
    ${sycl_lib}
    ${offline_flags}
    ${COMMON_DEVICE_LINK_FLAGS}
    SHARED
    SYCL_SOURCES ${sources})

  # Restore for subsequent non-ESIMD targets.
  set(SYCL_COMPILE_FLAGS ${_saved_flags})

  target_include_directories(${sycl_lib} PRIVATE ${_ESIMD_INCLUDE_DIRS})
  torch_compile_options(${sycl_lib})
  target_compile_options(${sycl_lib} PRIVATE ${TORCH_XPU_OPS_FLAGS})
  target_link_libraries(${sycl_lib}
    ${TORCH_LIBRARIES} c10 torch torch_cpu ${SYCL_LIBRARY})
  target_link_libraries(common_ops PUBLIC ${sycl_lib})
  install(TARGETS ${sycl_lib} LIBRARY DESTINATION sgl_kernel)
  set_target_properties(${sycl_lib} PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE)
endfunction()

# --- JIT bucket: one .so per source file (mirrors upstream convention) ---
foreach(src ${ESIMD_JIT_SRCS})
  get_filename_component(name ${src} NAME_WLE REALPATH)
  _add_esimd_lib(sgl-esimd-${name} "${_ESIMD_JIT_OFFLINE_FLAGS}" "${src}")
endforeach()

# --- AOT bucket ---
foreach(src ${ESIMD_AOT_SRCS})
  get_filename_component(name ${src} NAME_WLE REALPATH)
  _add_esimd_lib(sgl-esimd-${name} "${_ESIMD_AOT_OFFLINE_FLAGS}" "${src}")
endforeach()

# --- AOT + doubleGRF bucket ---
foreach(src ${ESIMD_LGRF_SRC})
  get_filename_component(name ${src} NAME_WLE REALPATH)
  _add_esimd_lib(sgl-esimd-${name} "${_ESIMD_LGRF_OFFLINE_FLAGS}" "${src}")
endforeach()
