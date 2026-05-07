# ATen XPU sources

file(GLOB device_cpp "sycl/*.cpp" "sycl/*.sycl")
file(GLOB host_cpp "./*.cpp" "./*.cc")

# BMG only
set(device_cpp_xe20)
# common SYCL kernels
set(device_cpp_common)

foreach(file ${device_cpp})
    get_filename_component(fname ${file} NAME)

    if(fname MATCHES "Xe20\\.(cpp|sycl)$")
        list(APPEND device_cpp_xe20 ${file})
    else()
        list(APPEND device_cpp_common ${file})
    endif()

endforeach()

include(FMHADecodeXe20.cmake)
include(FMHAPrefillXe20.cmake)
include(MlaDecodeXe20.cmake)

message(STATUS "BMG files: ${device_cpp_xe20}")
message(STATUS "Common files: ${device_cpp_common}")

list(APPEND ATen_XPU_CPP_SRCS ${host_cpp})
list(APPEND ATen_XPU_SYCL_COMMON ${device_cpp_common})
list(APPEND ATen_XPU_SYCL_XE20 ${device_cpp_xe20})

include(${CMAKE_CURRENT_SOURCE_DIR}/GroupGemmXe20.cmake)

set(ATen_XPU_CPP_SRCS ${ATen_XPU_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCL_COMMON ${ATen_XPU_SYCL_COMMON} PARENT_SCOPE)
set(ATen_XPU_SYCL_XE20 ${ATen_XPU_SYCL_XE20} PARENT_SCOPE)

include(BuildOnLinux.cmake)

# Vendored add-ons — each self-contained; upstream build rules untouched.
# GdnAttn re-enabled with cutlass pinned to upstream 2fc09973 (see top-level
# CMakeLists.txt comment). This combination is the safe variant: gdn_attn
# APIs we need already exist in 2fc09973; FMHA/MLA template expansion stays
# identical to upstream, so Qwen3.5 sampling path is unaffected.
include(${CMAKE_CURRENT_SOURCE_DIR}/GdnAttn.cmake)
# Esimd.cmake still disabled: with oneAPI 2025.3 the ESIMD reduce/sum template
# instantiation on simd<float, 2> fails to compile (invalid cast in
# esimd/detail/types.hpp, subscripted value is neither array nor pointer in
# intrin.hpp). Needs ESIMD header version alignment or kernel refactor
# (e.g. avoid detail::sum over length-2 vectors in qkv_split_norm_rope.h:107).
# include(${CMAKE_CURRENT_SOURCE_DIR}/Esimd.cmake)
