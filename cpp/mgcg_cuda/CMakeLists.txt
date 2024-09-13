cmake_minimum_required(VERSION 3.23)

set(CMAKE_CUDA_ARCHITECTURES 86;89)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# suppress the warning of unused function
# warning #177-D: function "xxx" was declared but never referenced 
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-diag-suppress=177")

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


## -------------------------------------------------------------------------- */
##                                optional targets                            */
## -------------------------------------------------------------------------- */

option(cppmgxpbd "Build cpp mgxpbd" OFF)
if(cppmgxpbd)
    add_executable(mgxpbd mgxpbd.cpp)
    target_include_directories(mgxpbd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)
endif()

option(test "Build test for debug" OFF)
if(test)
    add_executable(test_main test_main.cu)
    target_link_libraries(test_main PUBLIC fastmg)
endif()


# option(initFillCloth "cpp initFillCloth(OFF will use python initFill)" ON)
set(initFillCloth ON)
if(initFillCloth)
    # add_subdirectory(extern/pybind11)
    # pybind11_add_module(initFillCloth initFillCloth.cpp)
    target_sources(fastmg PRIVATE initFillCloth.cpp)
endif()