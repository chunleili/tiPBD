cmake_minimum_required(VERSION 3.23)

set(CMAKE_CUDA_ARCHITECTURES 86;89)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compile to mgcg_cuda/lib
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/lib")

project(fastmg LANGUAGES CXX CUDA)

add_library(fastmg SHARED fastmg.cu kernels.cuh)
target_link_libraries(fastmg PUBLIC cusparse cublas cusolver)
target_include_directories(fastmg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/eigen)
# set_target_properties(fastmg PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(mgxpbd mgxpbd.cpp)
# target_include_directories(mgxpbd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)

# add_executable(test_main test_main.cu)
# target_link_libraries(test_main PUBLIC fastmg)