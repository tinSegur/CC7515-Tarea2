#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

typedef unsigned char uchar;

__global__ void sim_life_shared(int n, int m, char *a, char *b);