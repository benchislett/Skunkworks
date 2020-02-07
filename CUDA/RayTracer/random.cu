#include "rt.cuh"

__device__ Vec3 random_in_unit_sphere(curandState *r)
{
  Vec3 p;
  Vec3 white = {1.0, 1.0, 1.0};
  do {
    p = (Vec3){curand_uniform(r), curand_uniform(r), curand_uniform(r)} * 2.0;
    p = p - white;
  } while (norm_sq(p) >= 1.0);
  return p;
}

__global__ void rand_init(const RenderParams p, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= p.width) || (j >= p.height)) return;

  int idx = i + p.width * j;
  curand_init(2020, idx, 0, &rand_state[idx]);
}
