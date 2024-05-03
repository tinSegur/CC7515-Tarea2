#include "utils.h"

void initGrid(int n, int m, uchar a[][]){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i][j] = rand() % 2;
        }
    }
}