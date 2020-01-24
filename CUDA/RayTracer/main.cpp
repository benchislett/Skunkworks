#include "rt.h"

#define WIDTH 256
#define HEIGHT 512

int main()
{
  size_t idx;
  int i,j;
  float output[3*WIDTH*HEIGHT];
  float r,g,b;

  render(output, WIDTH, HEIGHT);

  std::cout << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";

  for (j = 0; j < HEIGHT; j++)
  {
    for (i = 0; i < WIDTH; i++)
    {
      idx = 3 * (i + j * WIDTH);
      r = output[idx + 0];
      g = output[idx + 1];
      b = output[idx + 2];

      std::cout << (int)(255.99 * r) << " " << (int)(255.99 * g) << " " << (int)(255.99 * b) << std::endl;
    }
  }
}
