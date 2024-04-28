#include <OpenCL/opencl.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << CL_HPP_TARGET_OPENCL_VERSION << std::endl;
  return 0;
}
