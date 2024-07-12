cmake_minimum_required(VERSION 3.23)

set(CMAKE_CUDA_ARCHITECTURES 89)

project(spgemm LANGUAGES CXX CUDA)

add_executable(spgemm spgemm.cu)
target_link_libraries(spgemm PRIVATE cusparse cublas)