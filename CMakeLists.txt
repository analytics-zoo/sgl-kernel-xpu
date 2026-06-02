cmake_minimum_required(VERSION 3.19.2)
project(sgl_kernel)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)

set(ATen_XPU_SYCL_XE20)
set(ATen_XPU_SYCL_COMMON)

# Torch
find_package(Python3 COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

execute_process(
  COMMAND ${PYTHON3_EXECUTABLE}
          -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PY_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)


set(SGL_OPS_XPU_ROOT ${PROJECT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${SGL_OPS_XPU_ROOT}/cmake/Modules)

# PTL build attempt: build AOT GEN-ISA for Panther Lake (Xe3) in addition to
# bmg. Was FORCE-pinned to "bmg" upstream (B60-only). Allow override via the
# SGL_DPCPP_TARGET env var; default now targets PTL for this machine.
if(DEFINED ENV{SGL_DPCPP_TARGET})
  set(DPCPP_SYCL_TARGET "$ENV{SGL_DPCPP_TARGET}" CACHE STRING "dpcpp target device" FORCE)
else()
  set(DPCPP_SYCL_TARGET "ptl-h,ptl-u" CACHE STRING "dpcpp target device" FORCE)
endif()

include(${SGL_OPS_XPU_ROOT}/cmake/SYCL.cmake)
include(${SGL_OPS_XPU_ROOT}/cmake/BuildFlags.cmake)

include(FetchContent)

# SYCL support in cutlass
add_compile_definitions(CUTLASS_ENABLE_SYCL)
set(CUTLASS_ENABLE_SYCL ON CACHE BOOL "Enable SYCL in the cutlass" FORCE)
set(CUTLASS_ENABLE_BENCHMARKS  OFF CACHE BOOL "Remove benchmark to avoid cmake version issue in google benchmark" FORCE)
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable headers only mode in cutlass" FORCE)

# cutlass (upstream pin — DO NOT bump; bumping affects FMHA/MLA template
# expansion and breaks Qwen3.5 downstream sampling in this toolchain).
# gdn_attn/xe_2/ was originally written against 3f2a337e but the APIs it
# relies on (barrier_arrive, cutlass::get_sub_group_id, get_block_2d_copy_D,
# etc.) already exist in 2fc09973 — the earlier "needs 3f2a337e" diagnosis
# was wrong; the real build blocker was a missing -DSYCL_INTEL_TARGET.
FetchContent_Declare(
    repo-cutlass-sycl
    GIT_REPOSITORY https://github.com/intel/sycl-tla.git
    GIT_TAG        2fc09973bfdf15755090fcb0e3b6ad236408a992
    GIT_SHALLOW    OFF
)
FetchContent_MakeAvailable(repo-cutlass-sycl)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${repo-cutlass-sycl_SOURCE_DIR}/include
    ${repo-cutlass-sycl_SOURCE_DIR}/tools/util/include
)

add_subdirectory(${SGL_OPS_XPU_ROOT}/src)
