#include "kernel.h"

#include <cstring>
#include <future>

void sim_lifeCPU(size_t n, size_t m, char *grid, char *buf) {
    char liveNeighbors;

    size_t i0;
    size_t i2;

    size_t j0;
    size_t j2;

    for (size_t i = 0; i<n; i++) {
        i0 = (i-1)%n;
        i2 = (i+1)%n;

        for(size_t j = 0; j<m; j++) {
            j0 = (j-1)%m;
            j2 = (j+1)%m;

            liveNeighbors =
                    grid[i0*m + j0] + grid[i0*m + j] + grid[i0*m + j2] +
                    grid[i*m + j0] + grid[i*m + j2] +
                    grid[i2*m + j0] + grid[i2*m + j] + grid[i2*m + j2];

            buf[i*m + j] = liveNeighbors == 3 || (liveNeighbors == 2 && grid[i*m + j]);
        }
    }

    memcpy(grid, buf, n*m*sizeof(char));

}
