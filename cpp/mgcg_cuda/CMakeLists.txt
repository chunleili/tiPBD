cmake_minimum_required(VERSION 3.23)


# suppress the warning of unused function
# warning #177-D: function "xxx" was declared but never referenced 
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-diag-suppress=177")

set(CMAKE_CUDA_ARCHITECTURES 86;89)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compile to mgcg_cuda/lib
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/lib")
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/lib")

project(fastmg LANGUAGES CXX CUDA)

add_library(fastmg SHARED fastmg.cu cuda_utils.cu kernels.cu solve_soft.cu fastfill.cu cusparse_wrappers.cu smoother.cu vcycle.cu )
target_link_libraries(fastmg PUBLIC cusparse cublas cusolver)
target_include_directories(fastmg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen)
set_target_properties(fastmg PROPERTIES CUDA_SEPARABLE_COMPILATION ON
CUDA_STANDARD 17
CUDA_ARCHITECTURES "60;70;75;80")


add_executable(main main.cpp  meshio_tetgen.cpp physdata.cpp fastfill.cu kernels.cu linear_solver.cpp)
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen
)
add_subdirectory(extern/amgcl)
target_link_libraries(main PRIVATE  amgcl)
set(	OpenMP_CXX_FLAGS "")
set(	OpenMP_C_FLAGS "")
list(APPEND CUDA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
## -------------------------------------------------------------------------- */
##                                optional targets                            */
## -------------------------------------------------------------------------- */

option(cppmgxpbd "Build cpp mgxpbd" OFF)
if(cppmgxpbd)
    add_executable(mgxpbd mgxpbd.cpp)
    target_include_directories(mgxpbd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen)
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


set(graph_coloring ON)
if(graph_coloring)
    # add_executable(graph_coloring_main graph_coloring.cpp)
    target_sources(fastmg PRIVATE graph_coloring.cpp)
endif()


option(use_binding "Build python binding" OFF)
if(use_binding)
    add_subdirectory(extern/pybind)
    add_subdirectory(extern/eigen)
    add_subdirectory(binding)
endif()