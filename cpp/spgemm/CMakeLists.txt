cmake_minimum_required(VERSION 3.23)

set(CMAKE_CUDA_ARCHITECTURES 89)

project(spgemm LANGUAGES CXX CUDA)

add_library(spgemm SHARED spgemm.cu)
target_link_libraries(spgemm PRIVATE cusparse cublas)
set_property(TARGET spgemm PROPERTY POSITION_INDEPENDENT_CODE ON)