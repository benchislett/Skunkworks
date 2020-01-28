#ifndef BEN_RT_CUH
#define BEN_RT_CUH

// MISC

#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

#define EPSILON 0.000001

#define CMP_EPSILON 0.0001
#define EQ(a,b) (fabsf(a-b)<CMP_EPSILON)
#define ISZERO(a) (fabsf(a)<CMP_EPSILON)
#define SIGN(a) (a<0?-1:(a>0?1:0))

// VEC3

typedef struct {
  float x;
  float y;
  float z;
} Vec3;

__host__ __device__ Vec3 operator+(const Vec3 &a, const Vec3& b);
__host__ __device__ Vec3 operator-(const Vec3 &a, const Vec3 &b);
__host__ __device__ Vec3 operator*(const Vec3 &a, const Vec3 &b);
__host__ __device__ Vec3 operator*(const Vec3 &a, float x);
__host__ __device__ Vec3 operator/(const Vec3 &a, const Vec3 &b);
__host__ __device__ Vec3 operator/(const Vec3 &a, float x);

std::ostream& operator<<(std::ostream& os, const Vec3 &a);
__host__ __device__ bool operator==(const Vec3 &a, const Vec3 &b);

__host__ __device__ float norm(const Vec3 &a);
__host__ __device__ float norm_sq(const Vec3 &a);

__host__ __device__ float dot(const Vec3 &a, const Vec3 &b);
__host__ __device__ Vec3 cross(const Vec3 &a, const Vec3 &b);
__host__ __device__ Vec3 unit(const Vec3 &a);
__host__ __device__ void make_unit(Vec3 *a);

// RAY

typedef struct
{
  Vec3 from;
  Vec3 d;
} Ray;

__host__ __device__ Vec3 ray_at(const Ray &r, float t);

// HIT LOGIC

typedef struct {
  float time;
  Vec3 point;
  Vec3 normal;
} HitData;

// TRIANGLE SURFACE

typedef struct {
  Vec3 a;
  Vec3 b;
  Vec3 c;
} Tri;

__host__ __device__ bool hit(const Ray &r, const Tri &t, HitData *h);

// CAMERA

typedef struct {
  Vec3 location;
  Vec3 lower_left_corner;
  Vec3 horizontal;
  Vec3 vertical;
} Camera;

Camera make_camera(const Vec3 &location, const Vec3 &target, const Vec3 &view_up, float fov_vertical, float aspect);

__host__ __device__ Ray get_ray(const Camera &c, float u, float v);

// RENDER

typedef struct {
  int width;
  int height;
  Camera cam;
  Vec3 background;
  // World w;
} RenderParams;

void render(float *host_out, const Tri world, const RenderParams &p);

#endif
