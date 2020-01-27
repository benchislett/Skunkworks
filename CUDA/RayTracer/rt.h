#ifndef BEN_RT_H
#define BEN_RT_H

// MISC

#include <iostream>

#define EPSILON 0.0001
#define EQ(a,b) (fabsf(a-b)<EPSILON) 

// VEC3

typedef struct
{
  float x;
  float y;
  float z;
} Vec3;

Vec3 operator+(const Vec3 &a, const Vec3& b);
Vec3 operator-(const Vec3 &a, const Vec3 &b);
Vec3 operator*(const Vec3 &a, const Vec3 &b);
Vec3 operator*(const Vec3 &a, float x);
Vec3 operator/(const Vec3 &a, const Vec3 &b);
Vec3 operator/(const Vec3 &a, float x);

std::ostream& operator<<(std::ostream& os, const Vec3 &a);
bool operator==(const Vec3 &a, const Vec3 &b);

float norm(const Vec3 &a);
float norm_sq(const Vec3 &a);

float dot(const Vec3 &a, const Vec3 &b);
Vec3 cross(const Vec3 &a, const Vec3 &b);
Vec3 unit(const Vec3 &a);
void make_unit(Vec3 *a);

// RAY

typedef struct
{
  Vec3 from;
  Vec3 d;
} Ray;

Vec3 ray_at(const Ray &r, float t);

// RENDER

void render(float *host_out, int width, int height);

#endif
