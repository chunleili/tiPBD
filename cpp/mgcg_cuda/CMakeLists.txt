cmake_minimum_required(VERSION 3.23)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compile to mgcg_cuda/lib
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/lib")

project(my_project LANGUAGES CXX CUDA)

add_library(fast-vcycle-gpu SHARED vcycle.cu)
target_link_libraries(fast-vcycle-gpu PRIVATE cusparse cublas cusolver)
target_include_directories(fast-vcycle-gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)
set_target_properties(fast-vcycle-gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(mgxpbd mgxpbd.cpp)
# target_include_directories(mgxpbd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)