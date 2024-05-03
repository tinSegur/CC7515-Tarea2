kernel void sim_life(global int *a, int N, int M, int T) {
  uint dims = get_work_dim();

  int gindx = get_global_id(0);
  int gindy = get_global_id(1);


  if (gindex < N && gindy < M) {
    int t = T;
    while (t-- > 0){
        int x0 = (gindx - 1)%N;
        int x2 = (gindx + 1)%N;

        int y0 = (gindy - 1)%M;
        int y2 = (gindy + 1)%M;

        liveNeighbors = a[x0][y0] + a[x0][gindy] + a[x0][y2] +
                        a[gindx][y0] + a[gindx][y2] +
                        a[x2][y0] + a[x2][gindy] + a[x2][y2];

        barrier(CLK_GLOBAL_MEM_FENCE);
        a[gindx][gindy] = (liveNeighbors == 3 || (liveNeighbors == 2 && a[gindx][gindy])) ? 1 : 0;
    }
  }
}
