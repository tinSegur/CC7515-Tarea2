#include "kernel_shared.cuh"
#include <stdio.h>

__global__ void sim_life_shared(int n, int m, char *a, char *b){

    extern __shared__ char buf[];

    // get thread 2d id
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    // get shared memory dims
    int sn = blockDim.x + 2;
    int sm = blockDim.y + 2;

    // get shared memory id
    int si = threadIdx.x + 1;
    int sj = threadIdx.y + 1;

    int si0 = si - 1;
    int si2 = si + 1;

    int sj0 = sj - 1;
    int sj2 = sj + 1;

    // copy the corresponding value into shared memory
    // shared memory will be of size (blockDim.x + 2, blockDim.y + 2)
    // such that the values of threads within the block will be stored in the "center" of the shared memory array
    // the edges of the shared memory array will store values needed by the edge threads but that belong to another block's domain
    // so each thread must write its values translated diagonally in the positive direction
    // for example: thread (0,0) will require the value stored at (n,m)
    // given this, it will write the value at (0,0) into buffer memory (1,1) and write the value (n,m) into
    // buffer memory (0,0)
    // given edge values will still be accessed 3 times,
    // it's still better to copy the values into shared memory

    buf[si*sm + sj] = a[i*m + j];

    int i0 = (i + n - 1)%n;
    int i2 = (i + 1)%n;

    int j0 = (j + m - 1)%m;
    int j2 = (j + 1)%m;

    if (blockIdx.x == 0) { // threads in the upper horizontal block edge should write the value "above" them into shared memory
        buf[si0*sm + sj] = a[i0*m + j];

        if (blockIdx.y == 0) { // upper left corner needs to be copied as well
            buf[si0*sm + sj0] = a[i0*m + j0];
        }
        else if (blockIdx.y == blockDim.y-1) { // upper right
            buf[si0*sm + sj2] = a[i0*m + j2];
        }
    }
    else if (blockIdx.x == blockDim.x - 1) { //threads in the lower horizontal block edge should do the same with the value "below" them
        buf[si2*sm + sj] = a[i2*m + j];

        if (blockIdx.y == 0) { //lower left
            buf[si2*sm + sj0] = a[i2*m + j0];
        }
        else if (blockIdx.y == blockDim.y-1) { //lower right
            buf[si2*sm + sj2] = a[i2*m + j2];
        }
    }

    if (blockIdx.y == 0) { // threads in the left vertical block edge should write the value to the left into shared memory
        buf[si*sm + sj0] = a[i*m + j0];
    }
    else if (blockIdx.y == blockDim.y - 1) { // threads in the right vertical block edge should write the value to the right into shared memory
        buf[si*sm + sj2] = a[i*m + j2];
    }


    __syncthreads();

    char liveNeighbors;

    if (i < n && j < m) {
        liveNeighbors =
                buf[si0*sm + sj0] + buf[si0*sm + sj] + buf[si0*sm + sj2] +
                buf[si*sm + sj0]  + buf[si*sm + sj2] +
                buf[si2*sm + sj0] + buf[si2*sm + sj] + buf[si2*sm + sj2];
        b[i*m + j] = (liveNeighbors == 3 || (liveNeighbors == 2 && buf[si*m + sj]));
    }

}