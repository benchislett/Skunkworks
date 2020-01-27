#include "rt.h"

Vec3 operator+(const Vec3 &a, const Vec3 &b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 &a, const Vec3 &b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3 &a, const Vec3 &b)
{
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}

Vec3 operator*(const Vec3 &a, float x)
{
  return {x * a.x, x * a.y, x * a.z};
}

Vec3 operator/(const Vec3 &a, const Vec3 &b)
{
  return {a.x / b.x, a.y / b.y, a.z / b.z};
}

Vec3 operator/(const Vec3 &a, float x)
{
  return {a.x / x, a.y / x, a.z / x};
}

bool operator==(const Vec3 &a, const Vec3 &b)
{
  return EQ(a.x, b.x) && EQ(a.y, b.y) && EQ(a.z, b.z);
}

std::ostream& operator<<(std::ostream& os, const Vec3 &a)
{
  os << "Vec3(" << a.x << ", " << a.y << ", " << a.z << ")";
  return os;
}

Vec3 cross(const Vec3 &a, const Vec3 &b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float dot(const Vec3 &a, const Vec3 &b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

float norm_sq(const Vec3 &a)
{
  return dot(a, a);
}

float norm(const Vec3 &a)
{
  return sqrt(norm_sq(a));
}

Vec3 unit(const Vec3 &a)
{
  return a / norm(a);
}

void make_unit(Vec3 *a)
{
  float n = norm(*a);
  a->x /= n;
  a->y /= n;
  a->z /= n;
}
