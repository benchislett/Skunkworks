#include "rt.cuh"
#define MAX_DEPTH 1

__device__ Vec3 get_color(const Ray &r, const BVHWorld &w, const RenderParams &p, curandState *rand_state)
{
  Vec3 color = {1.0, 1.0, 1.0};
  const Vec3 white = {1.0, 1.0, 1.0};
  Vec3 tri_color = {0.9, 0.5, 0.7};
  HitData rec = {-1.0, white, white};
  Ray ray = r;
  int depth = 1;

  while (hit(ray, w, &rec)) {
    ray.from = rec.point;
    ray.d = rec.normal;
    color = color * tri_color;
    if (depth++ >= MAX_DEPTH) break;
  }

  float t = 0.5 * (unit(ray.d).y + 1);
  color = color * ((white * (1.0-t)) + (p.background * t));

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

__global__ void populate_bvh(Tri *t, BoundingNode *nodes, int n, int bn, int lower, int upper) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < lower || i >= upper) return;

  int left = 2 * i + 1;
  int right = 2 * i + 2;

  if (left < bn && right < bn) {
    nodes[i].left = &nodes[left];
    nodes[i].right = &nodes[right];
    nodes[i].slab = bounding_slab(nodes[left].slab, nodes[right].slab);
    nodes[i].t = NULL;
  } else if (left < bn) {
    nodes[i].left = &nodes[left];
    nodes[i].right = NULL;
    nodes[i].slab = nodes[left].slab;
    nodes[i].t = NULL;
  } else if (right < bn) {
    nodes[i].left = NULL;
    nodes[i].right = &nodes[right];
    nodes[i].slab = nodes[right].slab;
    nodes[i].t = NULL;
  } else {
    nodes[i].left = NULL;
    nodes[i].right = NULL;
    nodes[i].slab = bounding_slab(t[i - n]);
    nodes[i].t = &t[i - n];
  }
}

void render(float *host_out, const RenderParams &p, World w)
{
  int imgsize = 3 * p.width * p.height;

  float *device_out;
  cudaMalloc((void **)&device_out, imgsize * sizeof(float));

  Tri *device_tris;
  cudaMallocManaged((void **)&device_tris, w.n * sizeof(Tri));
  cudaMemcpy(device_tris, w.t, w.n * sizeof(Tri), cudaMemcpyHostToDevice);
  w.t = device_tris;

  BoundingNode *device_nodes;
  cudaMallocManaged((void **)&device_nodes, 2 * w.n * sizeof(BoundingNode));

  int tx = 8;
  int ty = 8;

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
}
