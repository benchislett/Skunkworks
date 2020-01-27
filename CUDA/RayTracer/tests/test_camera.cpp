#include "catch.hpp"

#include "../rt.h"

TEST_CASE( "Camera make_camera", "[Camera][make_camera]" ) {
  Vec3 location = {1.2, 3.1, 2.3};
  Vec3 target = {0.1, 0.01, 0.99};
  Vec3 view_up = {0.0, 1.0, 0.0};
  float fov_vertical = 40.0;
  float aspect = 1.50;

  Camera c = make_camera(location, target, view_up, fov_vertical, aspect);

  REQUIRE( c.location == location );
  REQUIRE( c.lower_left_corner == (Vec3){0.67521787, 2.0488322, 2.5240345} );
  REQUIRE( c.horizontal == (Vec3){0.8362069, 0.0, -0.7021585} );
  REQUIRE( c.vertical == (Vec3){-0.40953973, 0.35256082, -0.48772457} );
}

TEST_CASE( "Camera get_ray", "[Camera][get_ray]" ) {
  Camera c = {(Vec3){1.2, 3.1, 2.3}, (Vec3){0.554518, 1.8070636, 2.5755635}, (Vec3){1.0285345, 0.0, -0.863655}, (Vec3){-0.5037339, 0.4336498, -0.5999012}};

  Ray r = get_ray(c, 0.5, 0.65);

  REQUIRE( r.from == (Vec3){1.2, 3.1, 2.3} );
  REQUIRE( r.d == (Vec3){-0.458641785, -1.01106403, -0.54619978} );
}
