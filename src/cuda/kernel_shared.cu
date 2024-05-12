#include "kernel_shared.cuh"

__global__ void sim_life_shared(int n, int m, char *a, char *b){

    extern __shared__ char buf[];

    // get thread 2d id
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    // copy the corresponding value into shared memory
    // shared memory will be of size (blockDim.x + 2, blockDim.y + 2)
    // such that the values of threads within the block will be stored in the "center" of the shared memory array
    // the edges of the shared memory array will store values needed by the edge threads but that belong to another block's domain
    // so each thread must write its values translated diagonally in the positive direction
    // for example: thread (0,0) will require the value stored at (n,m)
    // given this, it will write the value at (0,0) into buffer memory (1,1) and write the value (n,m) into
    // buffer memory (0,0)
    // given edge values that are not in the corner will still be accessed 3 times,
    // it's still better to copy the values into shared memory

    buf[(blockIdx.x+1)*blockDim.y + blockIdx.y + 1] = a[i*m + j];

    int i0 = (i + n - 1)%n;
    int i2 = (i + 1)%n;

    int j0 = (j + m - 1)%m;
    int j2 = (j + 1)%m;

    if (blockIdx.x == 0) { // threads in the upper horizontal block edge should write the value "above" them into shared memory
        buf[blockIdx.y + 1] = a[i0*m + j];

        if (blockIdx.y == 0) { // upper left corner needs to be copied as well
            buf[0] = a[i0*m + j0];
        }
        else if (blockIdx.y == blockDim.y-1) { // upper right
            buf[blockDim.y+1] = a[i0*m + j2];
        }
    }
    else if (blockIdx.x == blockDim.x - 1) { //threads in the lower horizontal block edge should do the same with the value "below" them
        buf[(blockIdx.x+2)*blockDim.x + blockIdx.y + 1] = a[i2*m + j];

        if (blockIdx.y == 0) { //lower left
            buf[blockIdx.x*blockDim.x] = a[i0*m + j2];
        }
        else if (blockIdx.y == blockDim.y-1) { //lower right
            buf[(blockIdx.x + 2)*blockDim.x + blockDim.y + 1] = a[i2*m + j2];
        }

    }

    if (blockIdx.y == 0) { // threads in the left vertical block edge should write the value to the left into shared memory
        buf[(blockIdx.x+1)*blockDim.x] = a[i*m + j0];
    }
    else if (blockIdx.y == blockDim.y - 1) { // threads in the right vertical block edge should write the value to the right into shared memory
        buf[(blockIdx.x+1)*blockDim.x + blockIdx.y + 2] = a[i*m + j2];
    }

    int si0 = (blockIdx.x + blockDim.x - 1)%blockDim.x;
    int si = blockIdx.x;
    int si2 = (blockIdx.x + 1)%blockDim.x;

    int sj0 = (blockIdx.y + blockDim.y - 1)%blockDim.y;
    int sj = blockIdx.y;
    int sj2 = (blockIdx.y + 1)%blockDim.y;


    __syncthreads();

    char liveNeighbors;

    if (i < n && j < m)
        liveNeighbors =
                buf[si0*m + sj0] + buf[si0*m + sj] + buf[si0*m + sj2] +
                buf[si*m + sj0]  + buf[si*m + sj2] +
                buf[si2*m + sj0] + buf[si2*m + sj] + buf[si2*m + sj2];

    __syncthreads();

    if (i < n && j < m)
        b[i*m + j] = (liveNeighbors == 3 || (liveNeighbors == 2 && a[i*m + j]));


}