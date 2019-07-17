#define _POSIX_C_SOURCE 200809L

#include <cairo.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int width;
int height;
int num_triangles = 50;
unsigned int iterations = 100000;
unsigned int printSteps = 100;
double transparency = 0.5;

unsigned int **triangles;
double fitness = 0.0;
long int timestep = 1;

cairo_surface_t *target;

cairo_surface_t *surface;
cairo_t *cr;

struct timespec start;
struct timespec end;

void parse_args(int argc, char *argv[])
{
  int opt;
  while ((opt = getopt(argc, argv, "n:i:p:")) != -1)
  {
    switch (opt)
    {
    case 'n':
      num_triangles = atoi(optarg);
      break;
    case 'i':
      iterations = atoi(optarg);
      break;
    case 'p':
      printSteps = atoi(optarg);
      break;
    default:
      fprintf(stderr, "Usage: %s [-n numTriangles] [-i iterations] [-p printFrequency]\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

void new_triangle(int i)
{
  triangles[i][0] = rand() % width;  // x1
  triangles[i][1] = rand() % height; // y1
  triangles[i][2] = rand() % width;  // x2
  triangles[i][3] = rand() % height; // y2
  triangles[i][4] = rand() % width;  // x3
  triangles[i][5] = rand() % height; // y3
  triangles[i][6] = rand() % 256;    // r
  triangles[i][7] = rand() % 256;    // g
  triangles[i][8] = rand() % 256;    // b
}

void setup()
{
  srand(time(NULL));

  target = cairo_image_surface_create_from_png("data/target.png");

  width = cairo_image_surface_get_width(target);
  height = cairo_image_surface_get_height(target);

  surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, width, height);
  cr = cairo_create(surface);

  triangles = (unsigned int **)malloc(num_triangles * sizeof(unsigned int *));
  for (int i = 0; i < num_triangles; i++)
  {
    triangles[i] = (int *)malloc(9 * sizeof(unsigned int));
    new_triangle(i);
  }
}

void set_background_colour(unsigned char r, unsigned char g, unsigned char b)
{
  cairo_set_source_rgb(cr, (float)r / 255.0, (float)g / 255.0, (float)b / 255.0);
  cairo_rectangle(cr, 0, 0, width, height);
  cairo_fill(cr);
}

void draw_triangle(unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2, unsigned int x3, unsigned int y3, unsigned char r, unsigned char g, unsigned char b)
{
  cairo_set_source_rgba(cr, (float)r / 255.0, (float)g / 255.0, (float)b / 255.0, transparency);

  cairo_move_to(cr, x1, y1);
  cairo_line_to(cr, x2, y2);
  cairo_line_to(cr, x3, y3);
  cairo_close_path(cr);

  cairo_fill(cr);
}

void draw()
{
  set_background_colour(255, 255, 255);
  for (int i = 0; i < num_triangles; i++)
  {
    draw_triangle(triangles[i][0], triangles[i][1], triangles[i][2], triangles[i][3], triangles[i][4], triangles[i][5], triangles[i][6], triangles[i][7], triangles[i][8]);
  }
}

double get_fitness()
{
  double diff = 0;

  unsigned char *source_data = cairo_image_surface_get_data(surface);
  unsigned char *target_data = cairo_image_surface_get_data(target);

  for (int i = 0; i < width * height; i++)
  {
    target_data++;
    source_data++;
    diff += abs(*target_data++ - *source_data++);
    diff += abs(*target_data++ - *source_data++);
    diff += abs(*target_data++ - *source_data++);
  }

  return 100.0 * (1.0 - diff / ((double)width * (double)height * (double)(255.0 * 3.0)));
}

void step()
{

  unsigned int tmp;

  int idx = timestep % num_triangles;
  int x1, y1, x2, y2, x3, y3, r, g, b;

  x1 = triangles[idx][0];
  y1 = triangles[idx][1];
  x2 = triangles[idx][2];
  y2 = triangles[idx][3];
  x3 = triangles[idx][4];
  y3 = triangles[idx][5];
  r = triangles[idx][6];
  g = triangles[idx][7];
  b = triangles[idx][8];

  int mutate_color = rand() % 2;
  if (mutate_color)
  {
    triangles[idx][6] = rand() % 256;
    triangles[idx][7] = rand() % 256;
    triangles[idx][8] = rand() % 256;
  }
  else
  {
    int vertex = rand() % 3;
    triangles[idx][2 * vertex] = rand() % width;
    triangles[idx][2 * vertex + 1] = rand() % height;
  }
  draw();
  double new_fitness = get_fitness();

  if (new_fitness > fitness)
  {
    fitness = new_fitness;
    timestep++;
  }
  else
  {
    triangles[idx][0] = x1;
    triangles[idx][1] = y1;
    triangles[idx][2] = x2;
    triangles[idx][3] = y2;
    triangles[idx][4] = x3;
    triangles[idx][5] = y3;
    triangles[idx][6] = r;
    triangles[idx][7] = g;
    triangles[idx][8] = b;
  }
}

long time_diff(struct timespec a, struct timespec b)
{
  return (b.tv_nsec - a.tv_nsec < 0 ? 1000000000 : 0) + (b.tv_nsec - a.tv_nsec);
}

void teardown()
{
  cairo_destroy(cr);
  cairo_surface_write_to_png(surface, "output/output.png");
  cairo_surface_destroy(surface);
  cairo_surface_destroy(target);
}

int main(int argc, char *argv[])
{
  parse_args(argc, argv);

  setup();

  for (int i = 0; i < iterations; i++)
  {
    if (i % printSteps == 0)
    {
      clock_gettime(CLOCK_REALTIME, &end);
      printf("Iterations: %d, Improvements: %ld, Fitness: %f, Iterations/second: %ld\n", i, timestep, fitness, (long)(1000000000.0 * 100.0 / time_diff(start, end)));
      clock_gettime(CLOCK_REALTIME, &start);
    }
    step();
  }

  teardown();

  return 0;
}
