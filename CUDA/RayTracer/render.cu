#include "rt.cuh"
#include <curand_kernel.h>
#define MAX_DEPTH 2

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

__device__ Vec3 get_color(const Ray &r, const World &w, const RenderParams &p, curandState *rand_state)
{
  Vec3 color = {1.0, 1.0, 1.0};
  const Vec3 white = {1.0, 1.0, 1.0};
  Vec3 tri_color = {0.9, 0.5, 0.7};
  HitData rec = {-1.0, white, white};
  Ray ray = r;
  int depth = 1;

  while (hit(ray, w, &rec)) {
    ray.from = rec.point;
    ray.d = rec.normal;// + random_in_unit_sphere(rand_state);
    color = color * tri_color;
    if (depth++ >= MAX_DEPTH) break;
  }

  float t = 0.5 * (unit(ray.d).y + 1);
  color = color * ((white * (1.0-t)) + (p.background * t));

  return color;
}

__global__ void render_init(const RenderParams p, curandState *rand_state)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
 
  if ((i >= p.width) || (j >= p.height)) return;

  int idx = i + p.width * j;

  curand_init(2020, idx, 0, &rand_state[idx]);
}

__global__ void render_kernel(float *out, const World w, const RenderParams p, curandState *rand_state)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
 
  if ((i >= p.width) || (j >= p.height)) return;

  int idx = 3 * (i + p.width * j);
  float u = (float)i / (float)p.width;
  float v = (float)j / (float)p.height;

  curandState local_rand_state = rand_state[idx / 3];

  Ray r = get_ray(p.cam, u, v);
  Vec3 color = get_color(r, w, p, &local_rand_state);

  out[idx + 0] = color.x;
  out[idx + 1] = color.y;
  out[idx + 2] = color.z;
}

void render(float *host_out, const RenderParams &p, World w)
{
  float *device_out;
  cudaMalloc((void **)&device_out, 3 * p.width * p.height * sizeof(float));

  Tri *device_tris;
  cudaMalloc((void **)&device_tris, w.n * sizeof(Tri));
  cudaMemcpy(device_tris, w.t, w.n * sizeof(Tri), cudaMemcpyHostToDevice);
  w.t = device_tris;

  int tx = 8;
  int ty = 8;

  dim3 blocks(p.width / tx + 1, p.height / ty + 1);
  dim3 threads(tx, ty);

  curandState *rand_state;
  cudaMalloc((void **)&rand_state, 3 * p.width * p.height * sizeof(curandState));

  render_init<<<blocks, threads>>>(p, rand_state);
  render_kernel<<<blocks, threads>>>(device_out, w, p, rand_state);

  cudaDeviceSynchronize();

  cudaMemcpy(host_out, device_out, 3 * p.width * p.height * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_out);
  cudaFree(device_tris);
}
