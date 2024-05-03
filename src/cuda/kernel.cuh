
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

__global__ void sim_life(int n, int m, int a[n][m], int t);
