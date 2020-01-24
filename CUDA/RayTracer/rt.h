#ifndef BEN_RT_H
#define BEN_RT_H

#include <iostream>

typedef struct
{
  float x;
  float y;
  float z;
} Vec3;

Vec3 operator+(Vec3 a, Vec3 b);

#endif
