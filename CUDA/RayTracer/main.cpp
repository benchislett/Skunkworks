#include "rt.cuh"

#define WIDTH 256
#define HEIGHT 512

int main()
{
  size_t idx;
  int i,j;
  float output[3*WIDTH*HEIGHT];
  output[3 * WIDTH * HEIGHT - 1] = 0.123;
  float r,g,b;

  Camera c = make_camera((Vec3){1.0, 1.0, 1.0}, (Vec3){0.0, 0.0, 0.0}, (Vec3){0.0, 1.0, 0.0}, 40.0, (float)WIDTH / (float)HEIGHT);
  Vec3 background = {0.4, 0.4, 0.7};
  RenderParams p = {WIDTH, HEIGHT, c, background};

  render(output, p);

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
