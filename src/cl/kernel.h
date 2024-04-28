#ifndef CL_KERNEL_H
#define CL_KERNEL_H
#include <string>

#include <OpenCL/opencl.hpp>

class KernelLoader {
 public:
  KernelLoader(const std::string& source);

 private:
  cl::Device dev_;
  cl::Platform platf_;
  cl::CommandQueue queue_;
};

#endif  // CL_KERNEL_H
