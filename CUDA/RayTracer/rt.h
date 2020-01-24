#ifndef BEN_RT_H
#define BEN_RT_H

#include <iostream>

// VEC3

typedef struct
{
  float x;
  float y;
  float z;
} Vec3;

Vec3 operator+(Vec3 a, Vec3 b);
Vec3 operator-(Vec3 a, Vec3 b);
Vec3 operator*(Vec3 a, Vec3 b);
Vec3 operator/(Vec3 a, Vec3 b);

float norm(Vec3 *a);
float norm_sq(Vec3 *a);

float dot(Vec3 *a, Vec3 *b);
Vec3 cross(Vec3 *a, Vec3 *b);

// RAY


// RENDER

void render(float *host_out, int width, int height);

#endif
