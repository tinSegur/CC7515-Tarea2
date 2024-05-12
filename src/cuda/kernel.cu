#include "kernel.cuh"

__global__ void sim_life(int n, int m, char *a, char *b){
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i < n && j < m) {
        int i0 = (i + n - 1)%n;
        int i2 = (i + 1)%n;

        int j0 = (j + m - 1)%m;
        int j2 = (j + 1)%m;

        char liveNeighbors =
                a[i0*m + j0] + a[i0*m + j] + a[i0*m + j2] +
                a[i*m + j0]  + a[i*m + j2] +
                a[i2*m + j0] + a[i2*m + j] + a[i2*m + j2];

        __syncthreads();

        b[i*m + j] = (liveNeighbors == 3) || (liveNeighbors == 2 && a[i*m + j]) ? 1 : 0;
    }

}

