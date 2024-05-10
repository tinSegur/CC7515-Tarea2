enable_language(CUDA)
set(CUDA_SOURCES main.cu kernel.cu kernel.cuh kernel_shared.cu kernel_shared.cuh)
add_executable(${PROJECT_NAME}CUDA ${CUDA_SOURCES} ${UTILS})
target_include_directories(${PROJECT_NAME}CUDA PUBLIC ..)
