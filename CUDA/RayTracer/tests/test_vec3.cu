#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "../rt.cuh"

TEST_CASE( "Vec3 +", "[Vec3][plus]" ) {
  Vec3 a = {1.0, 1.234, 419.9};
  Vec3 b = {2.0, 3.0, 1.19};

  Vec3 expected = {3.0, 4.234, 421.09};

  REQUIRE( a + b == expected );
}

TEST_CASE( "Vec3 -", "[Vec3][minus]" ) {
  Vec3 a = {8.5734, 12.12110, -1.2323};
  Vec3 b = {-4.0, 6545.34, 763.5};

  Vec3 expected = {12.5734, -6533.2189, -764.7323};

  REQUIRE( a - b == expected );
}

TEST_CASE( "Vec3 * Vec3", "[Vec3][mul]" ) {
  Vec3 a = {1.0, 0.0, 3.5324};
  Vec3 b = {99.91, 0.33, 2.33};

  Vec3 expected = {99.91, 0.0, 8.230492};

  REQUIRE( a * b == expected );
}

TEST_CASE( "Vec3 * float", "[Vec3][mul]" ) {
  Vec3 a = {1.0, 0.0, 2.35342};
  float scale = 3.2;

  Vec3 expected = {3.2, 0.0, 7.530944};

  REQUIRE( a * scale == expected );
}

TEST_CASE( "Vec3 / Vec3", "[Vec3][div]" ) {
  Vec3 a = {0.0, 3.0, 6.0};
  Vec3 b = {5.3, 7.0, 11.2};

  Vec3 expected = {0.0 / 5.3, 3.0 / 7.0, 6.0 / 11.2};

  REQUIRE( a / b == expected );
}

TEST_CASE( "Vec3 / float", "[Vec3][div]" ) {
  Vec3 a = {1.0, 0.0, 2.35342};
  float scale = 3.2;

  Vec3 expected = {0.3125, 0.0, 0.73544375};

  REQUIRE( a / scale == expected );
}

TEST_CASE( "Vec3 cross", "[Vec3][cross]" ) {
  Vec3 a = {1.0, 2.3, 5.11};
  Vec3 b = {3.55, 4.31, 7.23};

  Vec3 expected = {-5.3951, 10.9105, -3.855};

  REQUIRE( cross(a, b) == expected );
}

TEST_CASE( "Vec3 dot", "[Vec3][dot]" ) {
  Vec3 a = {1.0, 64.32, 0.004};
  Vec3 b = {2.3, 5.11, 419.99};

  float expected = 332.655156;

  REQUIRE( dot(a, b) == Approx( expected ) );
}

TEST_CASE( "Vec3 norm_sq", "[Vec3][norm_sq]" ) {
  Vec3 a = {4.3, 2.4, 6.33};

  float expected = 64.3189;

  REQUIRE( norm_sq(a) == Approx( expected ) );
}

TEST_CASE( "Vec3 norm", "[Vec3][norm]" ) {
  Vec3 a = {4.3, 2.4, 6.33};

  float expected = 8.0199064832453;

  REQUIRE( norm(a) == Approx( expected ) );
}

TEST_CASE( "Vec3 unit", "[Vec3][unit]" ) {
  Vec3 a = {0.111, 2.0, 1.943};

  Vec3 expected = {0.0397760815236, 0.71668615357834, 0.6962605982};

  REQUIRE( unit(a) == expected );
}

TEST_CASE( "Vec3 make_unit", "[Vec3][make_unit]" ) {
  Vec3 a = {0.111, 2.0, 1.943};

  Vec3 expected = {0.0397760815236, 0.71668615357834, 0.6962605982};
  make_unit(&a);

  REQUIRE( a == expected );
}

