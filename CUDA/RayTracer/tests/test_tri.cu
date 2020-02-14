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
  REQUIRE( test_eq(rec.point, point) );
  REQUIRE( test_eq(rec.normal, normal) );
  REQUIRE( rec.time == Approx( 1.5 ) );
}

TEST_CASE( "World hit", "[World][Tri][hit]" ) {
  Tri t1 = {(Vec3){0.5, 0.5, 0.0}, (Vec3){0.75, 0.5, 0.0}, (Vec3){0.625, 0.625, 0.0}};
  Tri t2 = {(Vec3){0.5, 0.5, 1.0}, (Vec3){0.75, 0.5, 1.0}, (Vec3){0.625, 0.625, 1.0}};
  Tri t3 = {(Vec3){0.5, 0.5, -1.0}, (Vec3){0.75, 0.5, -1.0}, (Vec3){0.625, 0.625, -1.0}};

  Tri tris[3] = {t1, t2, t3};
  World w = {3, tris};

  Ray r = {(Vec3){0.625, 0.5487, 4.0}, (Vec3){0.0, 0.0, -1.2325}};

  HitData h1, h2, h3, h4;

  bool hit1 = hit(r, t1, &h1);
  bool hit2 = hit(r, t2, &h2);
  bool hit3 = hit(r, t3, &h3);

  bool hitAll = hit(r, w, &h4);

  Vec3 hit_pt = {0.625, 0.5487, 1.0};
  Vec3 normal = {0.0, 0.0, 1.0};

  REQUIRE( hitAll == true );
  REQUIRE( h4.time == Approx( 2.43408 ) );
  REQUIRE( test_eq(h4.point, hit_pt) );
  REQUIRE( test_eq(h4.normal, normal) );
}
