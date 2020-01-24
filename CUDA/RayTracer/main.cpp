#include "rt.h"

int main()
{
  Vec3 a = {1.0, 1.0, 1.0};
  Vec3 b = {2.0, -1.0, 3.0};

  Vec3 c = a + b;

  std::cout << c.x << " " << c.y << " " << c.z << std::endl;
}
