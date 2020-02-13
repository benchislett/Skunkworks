#include "rt.cuh"

__host__ __device__ bool hit(const Ray &r, const AABB &s, HitData *h)
{
  // Expanding components because this needs to be VERY fast
  float x = 1.0f / r.d.x;
  float y = 1.0f / r.d.y;
  float z = 1.0f / r.d.z;

  float t1 = (s.ll.x - r.from.x) * x;
  float t2 = (s.ur.x - r.from.x) * x;
  float t3 = (s.ll.y - r.from.y) * y;
  float t4 = (s.ur.y - r.from.y) * y;
  float t5 = (s.ll.z - r.from.z) * z;
  float t6 = (s.ur.z - r.from.z) * z;

  float tmin = MAX(MAX(MIN(t1, t2), MIN(t3, t4)), MIN(t5, t6));
  float tmax = MIN(MIN(MAX(t1, t2), MAX(t3, t4)), MAX(t5, t6));

  return (0 < tmax) && (tmin < tmax);
}

__host__ __device__ bool hit(const Ray &r, const BoundingNode &node, HitData *h) {
  if (node.left == NULL && node.right == NULL) return hit(r, *(node.t), h);
  if (node.left == NULL) return hit(r, *(node.right), h);
  if (node.right == NULL) return hit(r, *(node.left), h);

  if (hit(r, node.slab, h)) {
    HitData left_record, right_record;
    bool hit_left = hit(r, *(node.left), &left_record);
    bool hit_right = hit(r, *(node.right), &right_record);

    if (hit_left && hit_right) {
      if (left_record.time < right_record.time) *h = left_record;
      else *h = right_record;
      return true;
    } else if (hit_left) {
      *h = left_record;
      return true;
    } else if (hit_right) {
      *h = right_record;
      return true;
    }
    return false;
  }
  return false;
}

__host__ __device__ bool hit(const Ray &r, const BVHWorld &w, HitData *h) {
  return hit(r, w.nodes[0], h);
}

__host__ __device__ AABB bounding_slab(const Tri &t) {
  float xmin = MIN(MIN(t.a.x, t.b.x), t.c.x);
  float ymin = MIN(MIN(t.a.y, t.b.y), t.c.y);
  float zmin = MIN(MIN(t.a.z, t.b.z), t.c.z);

  float xmax = MAX(MAX(t.a.x, t.b.x), t.c.x);
  float ymax = MAX(MAX(t.a.y, t.b.y), t.c.y);
  float zmax = MAX(MAX(t.a.z, t.b.z), t.c.z);

  return {{xmin, ymin, zmin}, {xmax, ymax, zmax}};
}

__host__ __device__ AABB bounding_slab(const AABB &s1, const AABB &s2) {
  float ll_x = MIN(s1.ll.x, s2.ll.x);
  float ll_y = MIN(s1.ll.y, s2.ll.y);
  float ll_z = MIN(s1.ll.z, s2.ll.z);

  float ur_x = MAX(s1.ur.x, s2.ur.x);
  float ur_y = MAX(s1.ur.y, s2.ur.y);
  float ur_z = MAX(s1.ur.z, s2.ur.z);

  return {{ll_x, ll_y, ll_z}, {ur_x, ur_y, ur_z}};
}

__host__ __device__ float SA(const AABB &s) {
  float width = s.ur.x - s.ll.x;
  float height = s.ur.y - s.ll.y;
  float depth = s.ur.z - s.ll.z;
  
  return 2.0 * (width * height + height * depth + depth * width);
}
