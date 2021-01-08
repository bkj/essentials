cmake_minimum_required(VERSION 3.6)
project(essentials)

find_package(CUDA REQUIRED)

link_directories(
  /usr/local/cuda/lib64
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --generate-line-info")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DSM_TARGET=61") # ?? Should this be auto-detected?

set(ESSENTIALS_INCLUDE_DIR /home/ubuntu/projects/essentials CACHE STRING "Path to `gunrock/essentials` includes") # cafda937b683466484d0a6f0c3b721321cc98f59
cuda_add_library(essentials SHARED "${ESSENTIALS_INCLUDE_DIR}/externals/mtx/mmio.cpp")

target_include_directories(
  essentials
  PRIVATE
  "${ESSENTIALS_INCLUDE_DIR}/"
  "${ESSENTIALS_INCLUDE_DIR}/externals/moderngpu/src"
  "${ESSENTIALS_INCLUDE_DIR}/externals/rapidjson/include"
  "${ESSENTIALS_INCLUDE_DIR}/externals/mtx"
  "${TORCH_INCLUDE_DIRS}"
)

target_link_libraries(essentials ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} cudart)

set_target_properties(essentials PROPERTIES PREFIX "")