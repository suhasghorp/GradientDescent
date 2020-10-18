cmake_minimum_required(VERSION 3.16)
project(GradientDescent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(GD_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(GD_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(src)
add_subdirectory(Catch_tests)

