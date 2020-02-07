#include "catch.hpp"

#include "../rt.cuh"

TEST_CASE ( "AABB hit true", "[AABB][Ray][hit]" ) {
  Vec3 origin = {1.0, 1.0, 1.0};
  Vec3 direction = {2.0, -1.0, 3.0};

  Ray r = {origin, direction};

  Vec3 lower_left = {-100.0, -100.0, 10.0};
  Vec3 upper_right = {100.0, 100.0, 11.0};

  AABB slab = {lower_left, upper_right};

  HitData rec; // Unused

  REQUIRE ( hit(r, slab, &rec) ); 
}

TEST_CASE ( "AABB hit false", "[AABB][Ray][hit]" ) {
  Vec3 origin = {1.0, 1.0, 1.0};
  Vec3 direction = {2.0, -1.0, 3.0};

  Ray r = {origin, direction};

  Vec3 lower_left = {-100.0, -100.0, -10.0};
  Vec3 upper_right = {100.0, 100.0, -11.0};

  AABB slab = {lower_left, upper_right};

  HitData rec; // Unused

  REQUIRE ( !hit(r, slab, &rec) ); 
}

TEST_CASE( "AABB BoundingSlab", "[AABB][Tri][BoundingSlab]" ) {

}
