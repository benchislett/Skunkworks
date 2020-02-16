#include "rt.cuh"

#define WIDTH 4096
#define HEIGHT 4096
#define SAMPLES 16

World loadOFF(char *path)
{
  int i;
  unsigned long nverts, ntris, nfaces, n1, n2, n3;
  float v1, v2, v3;
  World w;
  Vec3 *verts;
  char buffer[1024];

  FILE *fp;
  fp = fopen(path, "r");

  fscanf(fp, "%s", &buffer);
  fscanf(fp, "%lu %lu %lu", &nverts, &ntris, &nfaces);

  verts = (Vec3 *)malloc(nverts * sizeof(Vec3));
  w.n = ntris;
  w.t = (Tri *)malloc(ntris * sizeof(Tri));

  for (i = 0; i < nverts; i++)
  {
    fscanf(fp, "%f %f %f", &v1, &v2, &v3);
    verts[i] = (Vec3){v1, v2, v3};
  }

  for (i = 0; i < ntris; i++)
  {
    fscanf(fp, "%lu %lu %lu %lu", &nfaces, &n1, &n2, &n3);
    w.t[i] = (Tri){verts[n1], verts[n2], verts[n3]};
  }

  fclose(fp);

  return w;
}

int main()
{
  size_t idx;
  int i,j;
  float *output = (float *)malloc(3 * WIDTH * HEIGHT * sizeof(float));
  float r,g,b;

  Camera c = make_camera((Vec3){0.0, 0.08, 0.3}, (Vec3){0.0, 0.08, 0.0}, (Vec3){0.0, -1.0, 0.0}, 40.0, (float)WIDTH / (float)HEIGHT);
  World w = loadOFF("data/bunny.off");
  Vec3 background = {0.4, 0.4, 0.7};
  RenderParams p = {WIDTH, HEIGHT, SAMPLES, c, background};

  render(output, p, w);
  free(w.t);

  std::cout << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";

  for (j = 0; j < HEIGHT; j++)
  {
    for (i = 0; i < WIDTH; i++)
    {
      idx = 3 * (i + j * WIDTH);
      r = sqrt(output[idx + 0]);
      g = sqrt(output[idx + 1]);
      b = sqrt(output[idx + 2]);

      std::cout << (int)(255.99 * r) << " " << (int)(255.99 * g) << " " << (int)(255.99 * b) << std::endl;
    }
  }
}
