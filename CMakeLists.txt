cmake_minimum_required(VERSION 3.6)
project(artOptimal)
find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O2 -Wno-deprecated-gpu-targets -use_fast_math " )
set(SOURCE_FILES artOptimal.cu)
cuda_add_executable(artOptimal ${SOURCE_FILES})

include_directories(/usr/local/cuda/lib64)
link_directories(/usr/lib/python2.7)
target_link_libraries (artOptimal python2.7)