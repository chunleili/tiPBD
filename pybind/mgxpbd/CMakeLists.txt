cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(fillA)

add_executable(fillA fillA.cpp)
target_include_directories(fillA PRIVATE eigen)