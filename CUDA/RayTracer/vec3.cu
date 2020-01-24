#include "rt.h"

Vec3 operator+(Vec3 a, Vec3 b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
