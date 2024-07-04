cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(rigid_body_modes)

add_subdirectory(pybind11)
# pybind11_add_module(amg_cg_solve_bind amg_cg_solve_bind.cpp)
pybind11_add_module(rigid_body_modes rigid_body_modes.cpp)



# add_custom_command(
#         TARGET ${PROJECT_NAME}  POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_directory
#         ${CMAKE_BINARY_DIR}/$<CONFIGURATION> ${CMAKE_CURRENT_LIST_DIR}
#         )