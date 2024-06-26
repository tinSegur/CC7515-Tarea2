#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <fstream>
#include <unistd.h>

#include "kernel.cuh"
#include "kernel_shared.cuh"

struct Times {
  long create_data;
  long copy_to_host;
  long execution;
  long copy_to_device;
  long copy_device_to_device;
  inline long total() {
    return create_data + copy_to_host + execution + copy_to_device + copy_device_to_device;
  }
};

Times t;

void initGrid(int n, int m, char *a){
  srand(1234);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      a[i*m + j] = rand() % 2;
    }
  }

}

bool simulate(int N, int M, int blockSize, int T = 50, std::string outfile = "cudaGameOfLife.txt", bool shared = false) {


  std::cout << "Starting simulation:\n";
  std::cout << "\tn: " << N << " m: " << M << " steps: " << T << " blockSize: " << blockSize << " shared?: " << (shared ? "yes" : "no") << "\n";


  using std::chrono::microseconds;
  std::size_t size = sizeof(char) * N * M;
  char *a = static_cast<char*>(malloc(size));

  // Create the memory buffers
  char *aDev;
  char *bDev;
  cudaMalloc(&aDev, size);
  cudaMalloc(&bDev, size);

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

  std::ofstream out;
  out.open(outfile);
  out << N << " " << M << " " << T << "\n";


  t.execution = 0;
  t.copy_to_host = 0;

  dim3 threadsPerBlock(blockSize, blockSize);
  dim3 gridDims(ceil(N/blockSize), ceil(M/blockSize));


  // Execute the function on the device (using 32 threads here)
  for (int i= T; i>0; i--) {
    t_start = std::chrono::high_resolution_clock::now();

    if (shared) {
      sim_life_shared<<<threadsPerBlock, gridDims, (blockSize+2)*(blockSize+2)*sizeof(char)>>>(N, M, aDev, bDev);
    }
    else {
      sim_life<<<threadsPerBlock, gridDims>>>(N, M, aDev, bDev);
    }

    cudaDeviceSynchronize();

    t_end = std::chrono::high_resolution_clock::now();
    t.execution +=
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

    t_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(aDev, bDev, size, cudaMemcpyDeviceToDevice); // bring aux buffer values back to main array
    t_end = std::chrono::high_resolution_clock::now();
    t.copy_device_to_device +=
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
            .count();

    // Copy the output variable from device to host
    t_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(a, aDev, size, cudaMemcpyDeviceToHost);
    t_end = std::chrono::high_resolution_clock::now();
    t.copy_to_host +=
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
            .count();

    out.write(a, size);

    //Print result
    for (int i = 0; i < N; i++){
      for (int j = 0; j < M; j++){
        int printerAux = a[i*M + j];
        out  << printerAux;
        if (j != M-1){
          out << ",";
        }
        else{
          out << "\n";
        }
      }
    }
    out << "\n";

  }

  cudaFree(aDev);
  cudaFree(bDev);
  delete a;


  std::cout << "cd:" << t.create_data << "\n";
  std::cout << "htd:" << t.copy_to_device << "\n";
  std::cout << "ke:" << t.execution << "\n";
  std::cout << "dth:" << t.copy_to_host << "\n";
  std::cout << "dtd:" << t.copy_device_to_device << "\n";
  std::cout << "tt:" << t.total();
  return true;

}

int main(int argc, char* argv[]) {
  if (argc != 7) {
    std::cerr << "Uso: " << argv[0] << " <n> <m> <sim steps> <block size> <use_shared> <output file>"
              << std::endl;
    return 2;
  }
  int n = std::stoi(argv[1]);
  int m = std::stoi(argv[2]);
  int steps = std::stoi(argv[3]);
  int bs = std::stoi(argv[4]);
  bool shared = bool(std::stoi(argv[5]));
  std::string outf = argv[6];

  if (!simulate(n, m, bs, steps, outf, shared)) {
    std::cerr << "CUDA: Error while executing the simulation" << std::endl;
    return 3;
    return 3;
  }

  std::ofstream out;
  out.open(argv[4], std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << argv[2] << "'" << std::endl;
    return 4;
  }
  /*// params
  out << n << "," << bs;
  // times
  out << t.create_data << "," << t.copy_to_device << "," << t.execution << "," << t.copy_to_host << "," << t.total() << "\n";

  std::cout << "Data written to " << argv[6] << std::endl;*/
  return 0;
}
