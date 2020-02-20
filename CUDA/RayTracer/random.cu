#include "rt.cuh"

__device__ Vec3 random_in_unit_sphere(curandState *r)
{
  float x,y,z;
  do {
    x = 2.0 * curand_uniform(r) - 1.0;
    y = 2.0 * curand_uniform(r) - 1.0;
    z = 2.0 * curand_uniform(r) - 1.0;
  } while (x*x + y*y + z*z >= 1.0);
  return {x, y, z};
}

__global__ void rand_init(const RenderParams p, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= p.width) || (j >= p.height)) return;

  int idx = i + p.width * j;
  curand_init(2020, idx, 0, &rand_state[idx]);
}
