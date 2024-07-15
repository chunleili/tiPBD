cmake_minimum_required(VERSION 3.23)

project(testLib LANGUAGES CXX)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_library(testLib SHARED testLib.cpp)
# position independent code
set_property(TARGET testLib PROPERTY POSITION_INDEPENDENT_CODE ON)