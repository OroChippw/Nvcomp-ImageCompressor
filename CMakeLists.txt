cmake_minimum_required(VERSION 3.0)
project(Nvcomp-ImageCompressor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    # Set the MSVC input code encoding format to utf-8
    set(CMAKE_C_FLAGS /source-charset:utf-8)
endif()

message(STATUS "Project: Nvcomp-ImageCompressor")
message(STATUS "Root Path: ${CMAKE_SOURCE_DIR}")

include(cmake/platform.cmake)  # checking platform

# link third-party library nvjpeg and opencv4
set(NVCOMP_DIR  ${CMAKE_SOURCE_DIR}/third_party/nvcomp_261_windows_12_x)
include_directories(${NVCOMP_DIR}/include)
include_directories(${NVCOMP_DIR}/include/nvcomp)
link_directories(${NVCOMP_DIR}/lib)
set(NVCOMP_LIB nvcomp nvcomp_bitcomp nvcomp_gdeflate)

set(CUDA_DIR  ${CMAKE_SOURCE_DIR}/third_party/cuda113)
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib)
set(CUDA_LIB cuda)

set(OPENCV_DIR  ${CMAKE_SOURCE_DIR}/third_party/opencv4)
include_directories(${OPENCV_DIR}/include)
link_directories("${OPENCV_DIR}/lib")
set(OPENCV_LIB opencv_world470d)
# set(OPENCV_LIB opencv_world470)

include_directories("src/ImageCompressorDll")

# compile own file
file(GLOB SRC_LIST 
    ${CMAKE_SOURCE_DIR}/src/ImageCompressor/*.cpp
    ${CMAKE_SOURCE_DIR}/src/ImageCompressorDll/*.cpp
)
add_executable(demo  ${SRC_LIST})

target_link_libraries(demo ${OPENCV_LIB} ${NVCOMP_LIB})

# Set dynamic library/static library generation path
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# # build static library [*.lib/*.a]
# add_library(nvcomp_compressor_static ${SRC_LIST})

# set_target_properties(
#     nvcomp_compressor_static PROPERTIES 
#     OUTPUT_NAME "nvcomp_compressor"
#     VERSION 1.0 SOVERSION 1 
#     DEFINE_SYMBOL NVCOMP_COMPRESS_RUNNER_EXPORTS
# )

# target_include_directories(
#     nvcomp_compressor_static PUBLIC
#     $<BUILD_INTERFACE:${OPENCV_DIR}/include>
#     $<BUILD_INTERFACE:${NVCOMP_DIR}/include>
# )

# target_link_libraries(
#     nvcomp_compressor_static PUBLIC
#     ${OPENCV_LIB} ${NVCOMP_LIB}
# )

# # build dynamic library [*.dll/*.so]
# add_library(nvcomp_compressor_dynamic SHARED ${SRC_LIST})

# set_target_properties(
#     nvcomp_compressor_dynamic PROPERTIES 
#     OUTPUT_NAME "nvcomp_compressor"
#     VERSION 1.0 SOVERSION 1 # dll version and api version
#     DEFINE_SYMBOL NVCOMP_COMPRESS_RUNNER_EXPORTS
# )

# target_include_directories(
#     nvcomp_compressor_dynamic PUBLIC
#     $<BUILD_INTERFACE:${OPENCV_DIR}/include>
#     $<BUILD_INTERFACE:${NVCOMP_DIR}/include>
# )

# target_link_libraries(
#     nvcomp_compressor_dynamic PUBLIC
#     ${OPENCV_LIB} ${NVCOMP_LIB}
# )
