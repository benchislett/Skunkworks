#include "rt.cuh"
// MOLLER-TRUMBORE
__host__ __device__ bool hit(const Ray &r, const Tri &t, HitData *h)
{
  Vec3 edge1 = t.b - t.a;
  Vec3 edge2 = t.c - t.a;

  Vec3 h_ = cross(r.d, edge2);

  float a = dot(edge1, h_);

  if (ISZERO(a)) return false;

  float f = 1.0 / a;
  Vec3 s = r.from - t.a;
  float u = f * dot(s, h_);

  if (u < 0.0 || u > 1.0) return false;

  Vec3 q = cross(s, edge1);
  float v = f * dot(r.d, q);
  
  if (v < 0.0 || u + v > 1.0) return false;

  float time = f * dot(edge2, q);

  if (time < 0.0001) return false;

  h->point = ray_at(r, time);
  h->time = time;
 
  float w = 1 - u - v;
  Vec3 normal = t.n_a * w + t.n_b * u + t.n_c * v;

  h->normal = normal * (-SIGN(dot(r.d, normal)));

  return true;
}

__host__ __device__ bool hit(const Ray &r, const World &w, HitData *h)
{
  bool did_hit = false;
  float closest = FLT_MAX;
  HitData closest_record;

  for (int i = 0; i < w.n; i++)
  {
    if (hit(r, w.t[i], h)) {
      if (h->time < closest) {
        did_hit = true;
        closest = h->time;
        closest_record = *h;
      }
    }
  }

  if (did_hit) {
    *h = closest_record;
    return true;
  } else {
    return false;
  }
}
