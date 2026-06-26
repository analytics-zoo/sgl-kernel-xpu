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
include(${CMAKE_CURRENT_SOURCE_DIR}/GroupGemmMxfp4W4A16Xe20.cmake)

set(ATen_XPU_CPP_SRCS ${ATen_XPU_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCL_COMMON ${ATen_XPU_SYCL_COMMON} PARENT_SCOPE)
set(ATen_XPU_SYCL_XE20 ${ATen_XPU_SYCL_XE20} PARENT_SCOPE)

include(BuildOnLinux.cmake)

# Vendored: GDN attention (from vllm-xpu-kernels via origin/dev). Requires
# common_ops target to exist, so must come after BuildOnLinux.cmake. Cherry-
# picked file-only; kept current dev-bmg cutlass pin (27bb6550) since the
# cutlass APIs gdn_attn uses (get_sub_group_id, barrier_arrive,
# get_block_2d_copy_A) already exist there.
include(${CMAKE_CURRENT_SOURCE_DIR}/GdnAttn.cmake)
