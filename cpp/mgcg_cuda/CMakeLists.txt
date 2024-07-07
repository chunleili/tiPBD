cmake_minimum_required(VERSION 3.23)

set(CMAKE_CUDA_ARCHITECTURES 89)

project(my_project LANGUAGES CXX CUDA)

add_library(fast-vcycle-gpu SHARED vcycle.cu)
target_link_libraries(fast-vcycle-gpu PRIVATE cusparse cublas)