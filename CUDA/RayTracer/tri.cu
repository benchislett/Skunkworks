#include "rt.cuh"

__host__ __device__ bool hit(const Ray &r, const Tri &t, HitData *h)
{
  Vec3 vertex1 = t.a;
  Vec3 vertex2 = t.b;
  Vec3 vertex3 = t.c;

  Vec3 edge1 = vertex2 - vertex1;
  Vec3 edge2 = vertex3 - vertex1;

  Vec3 h_ = cross(r.d, edge2);

  float a = dot(edge1, h_);

  if (ISZERO(a)) return false;

  float f = 1.0 / a;
  Vec3 s = r.from - vertex1;
  float u = f * dot(s, h_);

  if (u < 0.0 || u > 1.0) return false;

  Vec3 q = cross(s, edge1);
  float v = f * dot(r.d, q);
  
  if (v < 0.0 || u + v > 1.0) return false;

  float time = f * dot(edge2, q);

  if (time < EPSILON) return false;

  h->point = ray_at(r, time);
  h->time = time;
  
  Vec3 normal = unit(cross(edge1, edge2));

  h->normal = normal * (-SIGN(dot(r.d, normal)));

  return true;
}
