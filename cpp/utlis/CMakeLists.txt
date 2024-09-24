cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找 Python 3.10
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)

set (PYBIND11_PYTHON_VERSION "3.10")

project(pymgpbd)


add_subdirectory(extern/pybind11)


set(souces "")

option(use_rigid_body_modes "Use rigid_body_modes" OFF)
if(use_rigid_body_modes)
    list(APPEND sources rigid_body_modes.cpp)
endif()

# pybind11_add_module(init_adj init_adj.cpp)
# target_include_directories(init_adj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)

option(use_graph_coloring "Use graph_coloring" ON)
if(use_graph_coloring)
    # add_executable(graph_coloring_main graph_coloring.cpp)
    list(APPEND sources graph_coloring.cpp)
endif()

message(STATUS "sources: ${sources}")

pybind11_add_module(pymgpbd ${sources})
