#include "rt.cuh"
#define MAX_DEPTH 2

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

__global__ void populate_nodes(Tri *tris, BoundingNode *nodes, int n, int bn) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= bn) return;

  if (bn - i <= n) {
    nodes[i].slab = bounding_slab(tris[bn - i - 1]);
    nodes[i].t = (tris + bn - i - 1);
    nodes[i].left = NULL;
    nodes[i].right = NULL;
    return;
  }

  int left = 2 * i + 1;
  int right = 2 * i + 2;

  if (left < bn && right < bn) {
    nodes[i].t = NULL;
    nodes[i].left = (nodes + left);
    nodes[i].right = (nodes + right);
  } else if (left < bn) {
    nodes[i].t = NULL;
    nodes[i].left = (nodes + left);
    nodes[i].right = NULL;
  } else if (right < bn) {
    nodes[i].t = NULL;
    nodes[i].left = NULL;
    nodes[i].right = (nodes + right);
  }
}

__global__ void compute_aabbs(BoundingNode *nodes, int lower, int upper) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < lower || i >= upper) return;

  if (nodes[i].left == NULL && nodes[i].right == NULL) return;
  else if (nodes[i].left == NULL) {
    nodes[i].slab = nodes[i].right->slab;
  } else if (nodes[i].right == NULL) {
    nodes[i].slab = nodes[i].left->slab;
  } else {
    nodes[i].slab = bounding_slab(nodes[i].left->slab, nodes[i].right->slab);
  }
}

void render(float *host_out, const RenderParams &p, World w)
{
  int imgsize = 3 * p.width * p.height;

  float *device_out;
  cudaMalloc((void **)&device_out, imgsize * sizeof(float));

  Tri *device_tris;
  cudaMalloc((void **)&device_tris, w.n * sizeof(Tri));
  cudaMemcpy(device_tris, w.t, w.n * sizeof(Tri), cudaMemcpyHostToDevice);
  w.t = device_tris;

  BoundingNode *device_nodes;
  cudaMalloc((void **)&device_nodes, 2 * w.n * sizeof(BoundingNode));

  int tx = 8;
  int ty = 8;
  
  populate_nodes<<<2 * w.n / tx + 1, tx>>>(device_tris, device_nodes, w.n, 2 * w.n);

  int acc = w.n;
  while (acc > 0) {
    compute_aabbs<<<acc / 2 / tx + 1, tx>>>(device_nodes, acc / 2, acc);

    acc /= 2;
  }
  cudaDeviceSynchronize();

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
