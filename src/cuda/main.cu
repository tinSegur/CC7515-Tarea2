#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>
#include <iostream>
#include <fstream>

#include "kernel.cuh"
#include "../utils.h"

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

bool simulate(int N, int M, int blockSize, int gridSize, int T = 50) {
  using std::chrono::microseconds;
  std::size_t size = sizeof(uchar) * N * M;
  uchar a[N][M];

  // Create the memory buffers
  int **aDev;
  cudaMalloc(&aDev, size);

  // Assign values to host variables
  auto t_start = std::chrono::high_resolution_clock::now();
  initGrid(N, M, a);
  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(aDev, a, size, cudaMemcpyHostToDevice);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();


  // Execute the function on the device (using 32 threads here)
  t_start = std::chrono::high_resolution_clock::now();
  sim_life<<<blockSize, gridSize>>>(N, M, aDev, T);
  cudaDeviceSynchronize();
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Copy the output variable from device to host
  t_start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(a, aDev, size, cudaMemcpyDeviceToHost);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_host =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

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
  if (argc != 5) {
    std::cerr << "Uso: " << argv[0] << " <array size> <block size> <grid size> <output file>"
              << std::endl;
    return 2;
  }
  int n = std::stoi(argv[1]);
  int m = std::stoi(argv[2]);
  int gs = std::stoi(argv[3]);
  int bs = std::stoi(argv[4]);

  if (!simulate(n, m, bs, gs)) {
    std::cerr << "CUDA: Error while executing the simulation" << std::endl;
    return 3;
  }

  std::ofstream out;
  out.open(argv[4], std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << argv[2] << "'" << std::endl;
    return 4;
  }
  // params
  out << n << "," << bs << "," << gs << ",";
  // times
  out << t.create_data << "," << t.copy_to_device << "," << t.execution << "," << t.copy_to_host << "," << t.total() << "\n";

  std::cout << "Data written to " << argv[4] << std::endl;
  return 0;
}
