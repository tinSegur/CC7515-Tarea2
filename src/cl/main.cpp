
#include <cstddef>
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/opencl.hpp>
#endif  // DEBUG
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

struct Times {
  long create_data;
  long copy_to_host;
  long execution;
  long copy_to_device;
  inline long total() {
    return create_data + copy_to_host + execution + copy_to_device;
  }
};

Times t;
cl::Program prog;
cl::CommandQueue queue;

bool init() {
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  cl::Platform::get(&platforms);
  for (auto& p : platforms) {
    p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() > 0) break;
  }
  if (devices.size() == 0) {
    std::cerr << "Not GPU device found" << std::endl;
    return false;
  }

  std::cout << "GPU Used: " << devices.front().getInfo<CL_DEVICE_NAME>()
            << std::endl;

  cl::Context context(devices.front());
  queue = cl::CommandQueue(context, devices.front());

  std::ifstream sourceFile("kernel.cl");
  std::stringstream sourceCode;
  sourceCode << sourceFile.rdbuf();

  prog = cl::Program(context, sourceCode.str(), true);

  return true;
}

bool simulate(int N, int M, int localSize, int globalSize, int T = 50) {
  using std::chrono::microseconds;
  std::size_t size = sizeof(int) * N * M;
  int a[N][M];

  // Create the memory buffers
  cl::Buffer aBuff(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);

  // Assign values to host variables
  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      a[i][j] = std::rand() % 2;
    }
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  // usar CL_FALSE para hacerlo asíncrono
  queue.enqueueWriteBuffer(aBuff, CL_TRUE, 0, size, a);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Make kernel
  cl::Kernel kernel(prog, "sim_life");

  // Set the kernel arguments
  kernel.setArg(0, aBuff);
  kernel.setArg(1, N);
  kernel.setArg(2, M);
  kernel.setArg(3, T);

  // Execute the function on the device (using 32 threads here)
  cl::NDRange gSize(globalSize);
  cl::NDRange lSize(localSize);

  t_start = std::chrono::high_resolution_clock::now();
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
  queue.finish();
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Copy the output variable from device to host
  t_start = std::chrono::high_resolution_clock::now();
  queue.enqueueReadBuffer(aBuff, CL_TRUE, 0, size, a);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_host =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Print the result
  std::cout << "Time to create data: " << t.create_data << " microseconds\n";
  std::cout << "Time to copy data to device: " << t.copy_to_device
            << " microseconds\n";
  std::cout << "Time to execute kernel: " << t.execution << " microseconds\n";
  std::cout << "Time to copy data to host: " << t.copy_to_host
            << " microseconds\n";
  std::cout << "Time to execute the whole program: " << t.total()
            << " microseconds\n";
  return true;
}

int main(int argc, char* argv[]) {
  if (!init()) return 1;

  if (argc != 7) {
    std::cerr << "Uso: " << argv[0]
              << " <array size> <local size> <global size> <output file>"
              << std::endl;
    return 2;
  }
  int n = std::stoi(argv[1]);
  int m = std::stoi(argv[2]);
  int gs = std::stoi(argv[3]);
  int ls = std::stoi(argv[4]);

  if (!simulate(n, m, gs, ls)) {
    std::cerr << "CL: Error while executing the simulation" << std::endl;
    return 3;
  }

  std::ofstream out;
  out.open(argv[5], std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << argv[5] << "'" << std::endl;
    return 4;
  }
  // params
  out << n << "," << ls << "," << gs << ",";
  // times
  out << t.create_data << "," << t.copy_to_device << "," << t.execution << ","
      << t.copy_to_host << "," << t.total() << "\n";

  std::cout << "Data written to " << argv[4] << std::endl;
  return 0;
}
