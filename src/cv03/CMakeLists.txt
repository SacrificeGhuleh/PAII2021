#project(CudaTemplateProject LANGUAGES CXX CUDA)
project(CV03 CUDA)

add_executable(${PROJECT_NAME} runner.cu)
add_executable(${PROJECT_NAME}_globalMemory globalMemoryRunner.cu)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} PUBLIC ${COMMON_LIBS})
target_link_libraries(${PROJECT_NAME}_globalMemory PUBLIC ${COMMON_LIBS})
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)
target_compile_options(${PROJECT_NAME}_globalMemory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)
