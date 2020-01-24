#include "rt.h"

__global__ void render_kernel(float *out, int width, int height)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  
  if ((i >= width) || (j >= height)) return;

  int idx = 3 * (i + width * j);

  out[idx + 0] = 0.2;//(float)i / (float)width;
  out[idx + 1] = 0.2;//(float)j / (float)height;
  out[idx + 2] = 0.2;
}

void render(float *host_out, int width, int height)
{
  float *device_out;
  cudaMalloc(&device_out, 3 * width * height);

  int tx = 8;
  int ty = 8;

  dim3 blocks(width / tx + 1, height / ty + 1);
  dim3 threads(tx, ty);

  render_kernel<<<blocks, threads>>>(device_out, width, height);

  cudaMemcpy(host_out, device_out, 3 * width * height, cudaMemcpyDeviceToHost);

  cudaFree(device_out);
}
