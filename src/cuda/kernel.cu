#include "kernel.cuh"

__global__ void life_sim(int n, int m, int a[n][m], int t = 50){
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i < n && j < m) {
        while (t-- > 0) {
            int i0 = (i - 1)%n;
            int i2 = (i + 1)%n;

            int j0 = (j - 1)%m;
            int j2 = (j + 1)%m;

            char liveNeighbors = a[i0][j0] + a[i0][j] + a[i0][j2] +
                    a[i][j0]  + a[i][j2] +
                    a[i2][j0] + a[i2][j] + a[i2][j2];

            __syncthreads();

            a[i][j] = (liveNeighbors == 3) || (liveNeighbors == 2 && a[i][j]) ? 1 : 0;

        }
    }

}

