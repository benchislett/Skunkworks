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
  if (node.left == NULL && node.right == NULL) {
    return hit(r, *(node.t), h);
  }
  if (node.left == NULL) {
    return hit(r, *(node.right), h);
  }
  if (node.right == NULL) {
    return hit(r, *(node.left), h);
  }

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

__global__ void populate_morton_codes(Tri *t, uint64_t *codes, int n, AABB bounds, int res) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= n) return;

  codes[i] = mortonCode(t[i], bounds, res);
}
