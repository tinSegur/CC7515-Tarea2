enable_language(CUDA)
set(CUDA_SOURCES kernel.cu kernel.cuh kernel_shared.cu kernel_shared.cuh)
add_executable(${PROJECT_NAME}CUDA main.cu ${CUDA_SOURCES} ${UTILS})
target_include_directories(${PROJECT_NAME}CUDA PUBLIC ..)

add_executable(${PROJECT_NAME}CUDATEST main_test.cu ${CUDA_SOURCES} ../kernel.cpp ${UTILS})
target_include_directories(${PROJECT_NAME}CUDATEST PUBLIC ${PROJECT_SOURCE_DIR}/src)
