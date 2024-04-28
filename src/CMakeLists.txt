include(CheckLanguage)
find_package(OpenCL)
check_language(CUDA)
if(${CMAKE_CUDA_COMPILER})
  message("CUDA FOUND")
  add_subdirectory(cuda)
endif()
if(${OpenCL_FOUND})
  message("OpenCL FOUND")
  add_subdirectory(cl)
endif()


set(CPU_SOURCES main.cpp)
add_executable(${PROJECT_NAME}CPU ${CPU_SOURCES})

