# cmake_minimum_required(VERSION 3.15...3.26)
# project(pymgpbd LANGUAGES CXX)

# pybind11_add_module(pymgpbd
#   binding.cpp
#   ${CMAKE_SOURCE_DIR}/solve_soft.cu
# )

# set_target_properties(pymgpbd PROPERTIES
#         DEBUG_POSTFIX ""
#         RELWITHDEBINFO_POSTFIX ""
#         MINSIZEREL_POSTFIX "")
# target_include_directories(pymgpbd PUBLIC ${CMAKE_SOURCE_DIR})
# target_link_libraries(pymgpbd PRIVATE fastmg)
# add_dependencies(pymgpbd fastmg)

# add_custom_target(pymgpbd_install
#         ${PYTHON_EXECUTABLE} setup.py bdist_wheel
#         COMMAND ${PYTHON_EXECUTABLE} -m pip install -I dist/pymgpbd-0.0.1-cp310-cp310-win_amd64.whl
#         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# # install(TARGETS pymgpbd DESTINATION .)


# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX CUDA)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(_core MODULE    binding.cpp
   ${CMAKE_SOURCE_DIR}/solve_soft.cu WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers)
target_link_libraries(_core PRIVATE fastmg)
target_include_directories(_core PUBLIC ${CMAKE_SOURCE_DIR})
# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})