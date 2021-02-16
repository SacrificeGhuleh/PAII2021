cmake_minimum_required(VERSION 3.17)
project(ParallelAlgorithms2021 CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

#set(CMAKE_VERBOSE_MAKEFILE ON)


find_package(CUDAToolkit)

# Where to find CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Enable export of compile commands for PVS studio
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Setup path to libs
set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# Add embree to project
#include(${CMAKE_MODULE_PATH}/embree.cmake)
# Add glm to project
#include(${CMAKE_MODULE_PATH}/glm.cmake)
# Add glfw to project
#include(${CMAKE_MODULE_PATH}/glfw3.cmake)
# Add glad to project
#include(${CMAKE_MODULE_PATH}/glad.cmake)
# Add PVS studio to project
#include(${CMAKE_MODULE_PATH}/PVS-Studio.cmake)
# Add imgui to project
#include(${CMAKE_MODULE_PATH}/imgui.cmake)
# Add spdlog to project
#include(${CMAKE_MODULE_PATH}/spdlog.cmake)
# Add doxygen to project
#include(${CMAKE_MODULE_PATH}/doxygen.cmake)


include_directories(lib/pg)
include_directories(lib/utility)
include_directories(lib/cuda-samples/Common)

add_subdirectory(lib/utility)
#add_subdirectory(lib/cuda-samples/Samples/matrixMul)

set(COMMON_LIBS utility)

add_subdirectory(src)

