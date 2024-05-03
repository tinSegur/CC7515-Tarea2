enable_language(CUDA)
set(CUDA_SOURCES main.cu kernel.cu)
add_executable(${PROJECT_NAME}CUDA ${CUDA_SOURCES} ${UTILS})

