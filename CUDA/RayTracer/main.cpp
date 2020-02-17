#include "rt.cuh"

#define WIDTH 1024
#define HEIGHT 1024
#define SAMPLES 16

World loadOFF(const char *path)
{
  int i;
  World w;
  float x,y,z;
  int v1, v2, v3, n1, n2, n3;
  std::vector<Vec3> verts = {};
  std::vector<Vec3> normals = {};
  std::vector<Tri> tris = {};
  AABB bounds = {{0, 0, 0}, {0, 0, 0}};
  char buffer[1024];
  char buffer1[1024],buffer2[1024],buffer3[1024];

  FILE *fp;
  fp = fopen(path, "r");
  if (fp == NULL) {
    fprintf(stderr, "File not found!\n");
  }

  while (fgets(buffer, 1024, fp) != NULL) {
    if (buffer[0] == '#') continue;

    if (buffer[0] == 'v' && buffer[1] == ' ') {
      sscanf(buffer, "%*c %f %f %f", &x, &y, &z);
      verts.push_back((Vec3){x, y, z});
    }

    if (buffer[0] == 'v' && buffer[1] == 'n' && buffer[2] == ' ') {
      sscanf(buffer, "%*c%*c %f %f %f", &x, &y, &z);
      normals.push_back((Vec3){x,y,z});
    }

    if (buffer[0] == 'f' && buffer[1] == ' ') {
      Tri t;
      if (sscanf(buffer, "%*c %*d//%*d %*d//%*d %*d//%*d") != EOF) {
        sscanf(buffer, "%*c %d//%d %d//%d %d//%d", &v1, &n1, &v2, &n2, &v3, &n3);
        Vec3 a = verts[v1-1];
        Vec3 b = verts[v2-1];
        Vec3 c = verts[v3-1];
        Vec3 n_a = normals[n1-1];
        Vec3 n_b = normals[n2-1];
        Vec3 n_c = normals[n3-1];
        t = (Tri){a, b, c, n_a, n_b, n_c};
      } else {
        sscanf(buffer, "%*c %d %d %d", &v1, &v2, &v3);
        Vec3 a = verts[v1-1];
        Vec3 b = verts[v2-1];
        Vec3 c = verts[v3-1];
        Vec3 edge1 = b - a;
        Vec3 edge2 = c - a;
        Vec3 normal = unit(cross(edge1, edge2));
        t = (Tri){a, b, c, normal, normal, normal};
      }
      bounds = bounding_slab(bounds, bounding_slab(t));
      tris.push_back(t);
    }
  }

  fclose(fp);
  w.n = tris.size();
  w.t = (Tri *)malloc(w.n * sizeof(Tri));
  w.bounds = bounds;
  for (i = 0; i < w.n; i++) w.t[i] = tris[i];
  return w;
}

int main()
{
  size_t idx;
  int i,j;
  float *output = (float *)malloc(3 * WIDTH * HEIGHT * sizeof(float));
  float r,g,b;

  Camera c = make_camera((Vec3){0.0, 2.0, -4.0}, (Vec3){0.0, 1.0, 0.0}, (Vec3){0.0, -1.0, 0.0}, 40.0, (float)WIDTH / (float)HEIGHT);
  World w = loadOFF("data/armadillo.obj");
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
