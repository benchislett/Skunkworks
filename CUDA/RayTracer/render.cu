#include "rt.cuh"
#define MAX_DEPTH 16

inline __device__ Vec3 lerp(const Vec3 &a, const Vec3 &b, float factor) {
  return a * (1 - factor) + b * factor;
}

__device__ Vec3 get_color(const Ray &r, const BVHWorld &w, const RenderParams &p, curandState *rand_state)
{
  Vec3 color = {1.0, 1.0, 1.0};
  const Vec3 white = {1.0, 1.0, 1.0};
  HitData rec;
  Ray ray = r;
  int depth = 0;

  Vec3 attenuation = {0.75, 0.75, 0.75};

  while (hit(ray, w, &rec)) {
    ray.from = rec.point;
    ray.d = rec.normal + random_in_unit_sphere(rand_state);
    color = color * attenuation;
    if (depth++ >= MAX_DEPTH) {
      color = {0.0, 0.0, 0.0};
      break;
    }
  }

  float t = 0.5 * (unit(ray.d).y + 1);
  color = color * lerp(white, p.background, t);

  return color;
}

__global__ void render_kernel(float *out, const BVHWorld w, const RenderParams p, curandState *rand_state)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
 
  if ((i >= p.width) || (j >= p.height)) return;

  int idx = 3 * (i + p.width * j);
  curandState local_rand_state = rand_state[idx / 3];

  Vec3 color = {0.0, 0.0, 0.0};
  for (int c = 0; c < p.samples; c++)
  {
    float irand = (float)i + curand_uniform(&local_rand_state);
    float jrand = (float)j + curand_uniform(&local_rand_state);

    float u = irand / (float)p.width;
    float v = jrand / (float)p.height;

    Ray r = get_ray(p.cam, u, v);
    color = color + get_color(r, w, p, &local_rand_state);
  }
  color = color / (float)p.samples;

  out[idx + 0] = color.x;
  out[idx + 1] = color.y;
  out[idx + 2] = color.z;
}

void render(float *host_out, const RenderParams &p, World w)
{
  int imgsize = 3 * p.width * p.height;
  int tx = 16;
  int ty = 16;

  float *device_out;
  cudaMalloc((void **)&device_out, imgsize * sizeof(float));

  Tri *device_tris;
  cudaMallocManaged((void **)&device_tris, w.n * sizeof(Tri));
  cudaMemcpy(device_tris, w.t, w.n * sizeof(Tri), cudaMemcpyHostToDevice);

  uint64_t *morton_codes;
  cudaMallocManaged((void **)&morton_codes, w.n * sizeof(uint64_t));
  populate_morton_codes<<<w.n / tx + 1, tx>>>(device_tris, morton_codes, w.n, w.bounds, 2097151);

  thrust::sort_by_key(thrust::device, morton_codes, morton_codes + w.n, device_tris);

  BoundingNode *device_nodes;
  cudaMallocManaged((void **)&device_nodes, 2 * w.n * sizeof(BoundingNode));

  cudaDeviceSynchronize();

  int acc = 2 * w.n;
  while (acc > 0) {
    populate_bvh<<<acc / tx + 1, tx>>>(device_tris, device_nodes, w.n, 2 * w.n, acc / 2, acc);

    acc /= 2;
    cudaDeviceSynchronize();
  }

  BVHWorld bw = {w.n, 2 * w.n, device_nodes};

  dim3 blocks(p.width / tx + 1, p.height / ty + 1);
  dim3 threads(tx, ty);

  curandState *rand_state;
  cudaMalloc((void **)&rand_state, imgsize * sizeof(curandState));

  rand_init<<<blocks, threads>>>(p, rand_state);
  render_kernel<<<blocks, threads>>>(device_out, bw, p, rand_state);

  cudaDeviceSynchronize();

  cudaMemcpy(host_out, device_out, imgsize * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_out);
  cudaFree(device_tris);
  cudaFree(morton_codes);
}
