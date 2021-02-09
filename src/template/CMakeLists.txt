#project(CudaTemplateProject LANGUAGES CXX CUDA)
project(CudaTemplateProject LANGUAGES CUDA)

add_executable(${PROJECT_NAME}
               arrayUtils.cuh
               unifiedMemTest.cu)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} PUBLIC pg cuda-samples-common)
