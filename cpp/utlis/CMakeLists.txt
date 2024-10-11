cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# compile to bin
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/bin")
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/bin")
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/bin")
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/bin")
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/bin")
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/bin")

project(utils)

option(use_rigid_body_modes "Use rigid_body_modes" OFF)
option(use_pybind "Use pybind" OFF)
option(use_init_adj "Use init_adj" OFF)
option(use_graph_coloring "Use graph_coloring" OFF)
option(use_ecl "Use ecl coloring" ON)

if(use_pybind OR use_init_adj OR use_rigid_body_modes)
    # 查找 Python 3.10
    find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)
    set (PYBIND11_PYTHON_VERSION "3.10")

    include(ExternalProject)
    set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/extern)
    ExternalProject_Add(pybind
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    )
    ExternalProject_Add(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    )

    add_subdirectory(extern/pybind11)
    add_subdirectory(extern/eigen)

    pybind11_add_module(pymgpbd rigid_body_modes.cpp)
    pybind11_add_module(init_adj init_adj.cpp)
    target_include_directories(init_adj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)
endif()


if(use_graph_coloring)
    add_executable(graph_coloring graph_coloring.cpp)
endif()


if(use_ecl)
    enable_language(CUDA)
    add_executable(ECL-GC ECL-GC_12.cu)
    target_compile_options(ECL-GC PRIVATE -O3 -arch=native)
    add_executable(mm2ecl mm2ecl.cpp)
endif()