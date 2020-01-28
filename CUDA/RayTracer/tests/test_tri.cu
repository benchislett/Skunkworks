#include "catch.hpp"

#include "../rt.cuh"

TEST_CASE( "Tri hit", "[Tri][Ray][hit]" ) {
  Vec3 origin = {2.5, 1.75, 1.5};
  Vec3 direction = {-1.5, -1.0, -1.0};

  Ray r = {origin, direction};

  Vec3 vertex1 = {-0.5, -0.5, 0.0};
  Vec3 vertex2 = {0.5, 0.5, 0.0};
  Vec3 vertex3 = {0.25, 0.25, 0.25};

  Tri t = {vertex1, vertex2, vertex3};

  HitData rec;

  bool res = hit(r, t, &rec);

  Vec3 point = {0.25, 0.25, 0.0};
  Vec3 normal = {0.707107, -0.707107, 0.0};

  REQUIRE( res == true );
  REQUIRE( rec.point == point );
  REQUIRE( rec.normal == normal );
  REQUIRE( rec.time == Approx( 1.5 ) );
}
