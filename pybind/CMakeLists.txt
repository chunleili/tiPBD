cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(pymgpbd)

add_subdirectory(pybind11)

pybind11_add_module(rigid_body_modes rigid_body_modes.cpp)

pybind11_add_module(fillA fillA.cpp)
target_include_directories(fillA PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)