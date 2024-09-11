cmake_minimum_required(VERSION 3.19)

project(CUinteropGL CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "OFF")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/render)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/example)