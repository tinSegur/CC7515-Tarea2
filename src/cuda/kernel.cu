#include "kernel.cuh"

__global__ void vec_sum(int *a, int *b, int *c, int n){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

