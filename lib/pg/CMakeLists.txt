project(pg LANGUAGES CXX CUDA)

add_library(${PROJECT_NAME} INTERFACE)

target_sources(${PROJECT_NAME} INTERFACE
               cDefs.h
               cudaDefs.h
               imageManager.h
               multiplatformDefs.h)

target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
