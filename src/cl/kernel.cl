kernel void vec_sum(global int *a, global int *b, global int *c, int N) {
  int gindex = get_global_id(0);
  if (gindex < N) {
    c[gindex] = a[gindex] + b[gindex];
  }
}
