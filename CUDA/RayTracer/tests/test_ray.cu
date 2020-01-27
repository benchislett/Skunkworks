#include "catch.hpp"

#include "../rt.cuh"

TEST_CASE( "Ray ray_at", "[Ray][ray_at]" ) {
  Vec3 origin = {0.1, 0.001, 0.0001};
  Vec3 direction = {5.0, 3.9, 2.001};
  Ray r = {origin, direction};

  float time = 12.34;

  Vec3 expected = {61.8, 48.127, 24.69244};

  REQUIRE( ray_at(r, time) == expected );
}
