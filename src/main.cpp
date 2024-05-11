#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "kernel.h"
#include "utils.h"

struct Times {
  long create_data;
  long execution;

  long total() { return create_data + execution; }
};

Times t;

bool simulate(int N, int M, int T = 50, std::string outfile = "gameOfLifeCPU.txt") {
  using std::chrono::microseconds;
  char a[N*M], b[N*M];

  srand(1234);

  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      a[i*M + j] = rand()%2;
    }
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // File for storing of results
  //Formatted so first line contains simulation parameters (N, M, T), and the rest are the simulation results
  std::ofstream out;
  out.open(outfile);
  out << N << " " << M << " " << T << "\n";


  int steps = T;
  t.execution = 0;

  while (steps-- > 0) {
    t_start = std::chrono::high_resolution_clock::now();
    sim_lifeCPU(N, M, T, a, b);
    t_end = std::chrono::high_resolution_clock::now();
    t.execution +=
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
            .count();

    out.write(a, sizeof(char)*N*M);

  }

  // Print the result


  std::cout << "Time to create data: " << t.create_data << " microseconds\n";
  std::cout << "Time to execute kernel: " << t.execution << " microseconds\n";
  std::cout << "Time to execute the whole program: " << t.total()
            << " microseconds\n";

  return true;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << "Uso: " << argv[0] << " <array x> <array y> <output_file>"
              << std::endl;
    return 2;
  }

  int n = std::stoi(argv[1]);
  int m = std::stoi(argv[2]);
  int steps = std::stoi(argv[3]);
  std::string outf = argv[4];
  if (!simulate(n, m, steps, outf)) {
    std::cerr << "Error while executing the simulation" << std::endl;
    return 3;
  }

  std::ofstream out;
  out.open(argv[2], std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << argv[2] << "'" << std::endl;
    return 4;
  }
  out << n << "," << t.create_data << "," << t.execution << "," << t.total()
      << "\n";

  std::cout << "Data written to " << argv[4] << std::endl;
  return 0;
}
