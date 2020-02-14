#include "rt.cuh"

__host__ __device__ Vec3 operator+(const Vec3 &a, const Vec3 &b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ Vec3 operator-(const Vec3 &a, const Vec3 &b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ Vec3 operator*(const Vec3 &a, const Vec3 &b)
{
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ Vec3 operator*(const Vec3 &a, float x)
{
  return {x * a.x, x * a.y, x * a.z};
}

__host__ __device__ Vec3 operator/(const Vec3 &a, const Vec3 &b)
{
  return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__host__ __device__ Vec3 operator/(const Vec3 &a, float x)
{
  return {a.x / x, a.y / x, a.z / x};
}

__host__ __device__ bool operator==(const Vec3 &a, const Vec3 &b)
{
  return EQ(a.x, b.x) && EQ(a.y, b.y) && EQ(a.z, b.z);
}

__host__ __device__ bool test_eq(const Vec3 &a, const Vec3 &b) {
  return TEST_EQ(a.x, b.x) && TEST_EQ(a.y, b.y) && TEST_EQ(a.z, b.z);
}

std::ostream& operator<<(std::ostream& os, const Vec3 &a)
{
  os << "Vec3(" << a.x << ", " << a.y << ", " << a.z << ")";
  return os;
}

__host__ __device__ Vec3 cross(const Vec3 &a, const Vec3 &b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__host__ __device__ float dot(const Vec3 &a, const Vec3 &b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float norm_sq(const Vec3 &a)
{
  return dot(a, a);
}

__host__ __device__ float norm(const Vec3 &a)
{
  return sqrt(norm_sq(a));
}

__host__ __device__ Vec3 unit(const Vec3 &a)
{
  return a / norm(a);
}

__host__ __device__ void make_unit(Vec3 *a)
{
  float n = norm(*a);
  a->x /= n;
  a->y /= n;
  a->z /= n;
}
