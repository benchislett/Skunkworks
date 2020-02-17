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

void populate_bvh_cpu(Tri *t, BoundingNode *nodes, int n, int bn) {
  int left, right;
  for (int i = bn - 1; i >= 0; i--) {
    left = 2 * i + 1;
    right = 2 * i + 2;

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
}

__host__ __device__ inline uint64_t split3(uint32_t a) {
  uint64_t x = a & 0x1fffff;
  x = (x | x << 32) & 0x1f00000000ffff;
  x = (x | x << 16) & 0x1f0000ff0000ff;
  x = (x | x << 8)  & 0x100f00f00f00f00f;
  x = (x | x << 4)  & 0x10c30c30c30c30c3;
  x = (x | x << 2)  & 0x1249249249249249;
  return x;
}

__host__ __device__ uint64_t mortonCode(const Tri &t, const AABB &bounds, int res) {
  float x,y,z;
  x = (t.a.x + t.b.x + t.c.x) / 3.0;
  y = (t.a.y + t.b.y + t.c.y) / 3.0;
  z = (t.a.z + t.b.z + t.c.z) / 3.0;
  float xlen,ylen,zlen;
  xlen = bounds.ur.x - bounds.ll.x;
  ylen = bounds.ur.y - bounds.ll.y;
  zlen = bounds.ur.z - bounds.ll.z;
  uint32_t a = res * ((x - bounds.ll.x) / xlen);
  uint32_t b = res * ((y - bounds.ll.y) / ylen);
  uint32_t c = res * ((z - bounds.ll.z) / zlen);

  uint64_t code = 0;
  code |= split3(a) | (split3(b) << 1) | (split3(c) << 2);

  return code;
}

__global__ void compute_morton_codes(Tri *t, uint64_t *codes, int n, AABB bounds, int res) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= n) return;

  codes[i] = mortonCode(t[i], bounds, res);
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
  compute_morton_codes<<<w.n / tx + 1, tx>>>(device_tris, morton_codes, w.n, w.bounds, 2097151);

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
}
