#include "kernel.h"

#include <future>

void sim_lifeCPU(size_t n, size_t m, int t, uchar grid[n][m], uchar buf[n][m]) {
    uchar liveNeighbors;
    int time = t;

    size_t i0;
    size_t i2;

    size_t j0;
    size_t j2;

    while (time-- > 0) {
        for (size_t i = 0; i<n; i++) {
            i0 = (i-1)%n;
            i2 = (i+1)%n;

            for(size_t j = 0; j<m; j++) {
                j0 = (j-1)%m;
                j2 = (j+1)%m;

                liveNeighbors = grid[i0][j0] + grid[i0][j] + grid[i0][j2] +
                        grid[i][j0] + grid[i][j2] +
                        grid[i2][j0] + grid[i2][j] + grid[i2][j2];

                buf[i][j] = (liveNeighbors == 3) || (liveNeighbors == 2 && grid[i][j]) ? 1 : 0;
            }
        }
        std::swap(grid, buf);
    }
}
