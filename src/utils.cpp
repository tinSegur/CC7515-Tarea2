#include "utils.h"

void initGrid(int n, int m, char *a){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i*m + j] = rand() % 2;
        }
    }
}