#include "rt.cuh"

__device__ Vec3 get_color(const Ray &r, const Vec3 &background)
{
  return background;
}

__global__ void render_kernel(float *out, const RenderParams p)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
 
  if ((i >= p.width) || (j >= p.height)) return;

  int idx = 3 * (i + p.width * j);
  float u = (float)i / (float)p.width;
  float v = (float)j / (float)p.height;

  Ray r = get_ray(p.cam, u, v);
  Vec3 color = get_color(r, p.background);

  out[idx + 0] = color.x;
  out[idx + 1] = color.y;
  out[idx + 2] = color.z;
}

void render(float *host_out, const RenderParams &p)
{
  float *device_out;
  cudaMalloc((void **)&device_out, 3 * p.width * p.height * sizeof(float));

  int tx = 8;
  int ty = 8;

  dim3 blocks(p.width / tx + 1, p.height / ty + 1);
  dim3 threads(tx, ty);

  render_kernel<<<blocks, threads>>>(device_out, p);

  cudaDeviceSynchronize();

  cudaMemcpy(host_out, device_out, 3 * p.width * p.height * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_out);
}
