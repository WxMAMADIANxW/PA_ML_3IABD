cmake_minimum_required(VERSION 3.19)
project(CPPDLLForPython)

set(CMAKE_CXX_STANDARD 17)

add_library(CPPDLLForPython SHARED library.cpp library.h)
include_directories("${PROJECT_SOURCE_DIR}/Eigen")
target_link_libraries(CPPDLLForPython ${PROJECT_SOURCE_DIR}/Eigen)
