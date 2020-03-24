#pragma once

#include <math.h>

float lerp(t, a, b) { return a + t * (b - a); }

struct Vec3 {
  float e[3];
  
  Vec3() { e[0] = 0.f; e[1] = 0.f; e[2] = 0.f; }
  Vec3(float w) { e[0] = w; e[1] = w; e[2] = w; }
  Vec3(float x, float y, float z) { e[0] = x; e[1] = y; e[2] = z; }

  Vec3(const Vec3 &v) { e[0] = v.e[0]; e[1] = v.e[1]; e[2] = v.e[2]; }
  Vec3 &operator=(const Vec3 &v) { e[0] = v.e[0]; e[1] = v.e[1]; e[2] = v.e[2]; return *this; }

  Vec3 operator+(const Vec3 &v) const { return Vec3(e[0] + v.e[0], e[1] + v.e[1], e[2] + v.e[2]); }
  Vec3 &operator+=(const Vec3 &v) { e[0] += v.e[0]; e[1] += v.e[1]; e[2] += v.e[2]; return *this; }

  Vec3 operator-(const Vec3 &v) const { return Vec3(e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2]); }
  Vec3 &operator-=(const Vec3 &v) { e[0] -= v.e[0]; e[1] -= v.e[1]; e[2] -= v.e[2]; return *this; }

  bool operator==(const Vec3 &v) const { return e[0] == v.e[0] && e[1] == v.e[1] && e[2] == v.e[2]; }
  bool operator!=(const Vec3 &v) const { return e[0] != v.e[0] || e[1] != v.e[1] || e[2] != v.e[2]; }

  Vec3 operator*(const float f) const { return Vec3(e[0] * f, e[1] * f, e[2] * f); }
  Vec3 &operator*=(const float f) { e[0] *= f; e[1] *= f; e[2] *= f; return *this; }

  Vec3 operator/(const float f) const { (*this) * (1.f / f); }
  Vec3 &operator/=(const float f) { return (*this) *= (1.f / f); }
};

bool is_zero(const Vec3 &v) { return v.e[0] == 0.f && v.e[1] == 0.f && v.e[2] == 0.f; }
bool has_nans(const Vec3 &v) { return std::isnan(v.e[0]) || std::isnan(v.e[1]) || std::isnan(v.e[2]); }

float length_sq(const Vec3 &v) { return v.e[0] * v.e[0] + v.e[1] * v.e[1] + v.e[2] * v.e[2]; }
float length(const Vec3 &v) { return std::sqrtf(length_sq(v)); }

float dot(const Vec3 &v1, const Vec3 &v2) { return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2]; }
float dot_abs(const Vec3 &v1, const Vec3 &v2) { return std::fabs(dot(v1, v2)); }

Vec3 abs(const Vec3 &v) { return Vec3(std::fabs(v.e[0]), std::fabs(v.e[1]), std::fabs(v.e[2])); }

Vec3 lerp(const float t, const Vec3 &v1, const Vec3 &v2) {
  return Vec3(lerp(t, v1.e[0], v2.e[0]), lerp(t, v1.e[1], v2.e[1]), lerp(t, v1.e[2], v2.e[2]));
}


