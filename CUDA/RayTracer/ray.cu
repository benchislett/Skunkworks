#include "rt.cuh"

__host__ __device__ Vec3 ray_at(const Ray &r, float t)
{
  Vec3 out = r.from + (r.d * t);
  return out;
}
