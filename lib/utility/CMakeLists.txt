cmake_minimum_required(VERSION 3.17)
project(utility LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

add_library(${PROJECT_NAME} format.h)

target_sources(${PROJECT_NAME} PRIVATE
               rng.h
               rng.cpp
               timer.h
               timer.cpp)

target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_LIST_DIR})
