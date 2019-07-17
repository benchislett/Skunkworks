// Enable some functionality for cairo
#define _POSIX_C_SOURCE 200809L

#include <cairo.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Width and Height are determined dynamically
int width;
int height;

// Arguments with defaults
int num_triangles = 50;
unsigned int iterations = 100000;
unsigned int printSteps = 100;

// Alpha level of a triangle, from 0 to 1
const double transparency = 0.5;

// Triangles matrix, an array of length num_triangles,
// with each element having 9 values: the 3 (x, y) coordinates, and an (r, g, b) colour
unsigned int **triangles;

// Initialize some global counters
double fitness = 0.0;
long int timestep = 1;

// Surface for the target image
cairo_surface_t *target;

// Surface for the main image, and the cairo instance
cairo_surface_t *surface;
cairo_t *cr;

struct timespec start;
struct timespec end;

// Parse the command arguments into the global variables
void parse_args(int argc, char *argv[])
{
  int opt;
  while ((opt = getopt(argc, argv, "n:i:p:")) != -1)
  {
    switch (opt)
    {
    case 'n':
      // Number of triangles to use for the simulation
      num_triangles = atoi(optarg);
      break;
    case 'i':
      // Number of iterations to evolve the image
      iterations = atoi(optarg);
      break;
    case 'p':
      // Number of steps between each printing of fitness to stdout
      printSteps = atoi(optarg);
      break;
    default:
      // Print a help script if invalid arguments are entered
      fprintf(stderr, "Usage: %s [-n numTriangles] [-i iterations] [-p printFrequency]\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

// Regenerate the ith triangle in the global array
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

// Initialize the global variables
void setup()
{
  // Random seed for new results: disable when debugging
  srand(time(NULL));

  // Load the target image into a cairo_surface_t
  target = cairo_image_surface_create_from_png("data/target.png");

  // Determine and save the dimensions of the target image
  width = cairo_image_surface_get_width(target);
  height = cairo_image_surface_get_height(target);

  // Create the context for the main image
  surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, width, height);
  cr = cairo_create(surface);

  // Allocate memory for the global triangles array
  triangles = (unsigned int **)malloc(num_triangles * sizeof(unsigned int *));
  for (int i = 0; i < num_triangles; i++)
  {
    triangles[i] = (int *)malloc(9 * sizeof(unsigned int));
    // Initialize the triangles to random values
    new_triangle(i);
  }
}

// Set the background colour by drawing a rectangle with the dimensions of the canvas
void set_background_colour(unsigned char r, unsigned char g, unsigned char b)
{
  cairo_set_source_rgb(cr, (float)r / 255.0, (float)g / 255.0, (float)b / 255.0);
  cairo_rectangle(cr, 0, 0, width, height);
  cairo_fill(cr);
}

// Draw a triangle given the 3 (x, y) coordinates and the (r, g, b) colour
void draw_triangle(unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2, unsigned int x3, unsigned int y3, unsigned char r, unsigned char g, unsigned char b)
{
  // Set the colour with the global transparency constant
  cairo_set_source_rgba(cr, (float)r / 255.0, (float)g / 255.0, (float)b / 255.0, transparency);

  cairo_move_to(cr, x1, y1);
  cairo_line_to(cr, x2, y2);
  cairo_line_to(cr, x3, y3);
  cairo_close_path(cr);

  cairo_fill(cr);
}

// Fix the background colour and redraw each of the triangles
void draw()
{
  set_background_colour(255, 255, 255);
  for (int i = 0; i < num_triangles; i++)
  {
    draw_triangle(triangles[i][0], triangles[i][1], triangles[i][2], triangles[i][3], triangles[i][4], triangles[i][5], triangles[i][6], triangles[i][7], triangles[i][8]);
  }
}

// Evaluate the raw percent difference between the two images
double get_fitness()
{
  double diff = 0;

  // Get a pointer to the raw pixel data of the image
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

  // Normalize the difference by the maximum difference, invert, and multiply by 100 to get the percent value
  return 100.0 * (1.0 - diff / ((double)width * (double)height * (double)(255.0 * 3.0)));
}

// Iterate the evolution process
void step()
{
  // Get the index of the current triangle
  int idx = timestep % num_triangles;

  // Save each of the triangle's values before evolving
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

  // Decide to mutate to either the position or colour with even chance
  int mutate_color = rand() % 2;
  if (mutate_color)
  {
    // Randomize the triangle's colour
    triangles[idx][6] = rand() % 256;
    triangles[idx][7] = rand() % 256;
    triangles[idx][8] = rand() % 256;
  }
  else
  {
    // Decide which vertex to mutate, with even chance
    int vertex = rand() % 3;
    // Double the vertex index for the x and y coordinate
    triangles[idx][2 * vertex] = rand() % width;
    triangles[idx][2 * vertex + 1] = rand() % height;
  }

  // Draw onto the canvas and use it to evaluate the similarity to the target
  draw();
  double new_fitness = get_fitness();

  if (new_fitness > fitness)
  {
    // If the new image is more fit than the unchanged, save the new triangle
    fitness = new_fitness;
    timestep++;
  }
  else
  {
    // If the new image is less fit than the unchanged, reset to the previous state
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

// Evalutate the difference between two timespec instances in seconds
double time_diff(struct timespec a, struct timespec b)
{
  // Get the second difference and add the scaled nanosecond difference
  double diff = (double)(end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) / 1000000000.0);
  return diff;
}

// Destroy, deallocate, and conclude operations
void teardown()
{
  // Destroy cairo objects
  cairo_destroy(cr);
  cairo_surface_write_to_png(surface, "output/output.png");
  cairo_surface_destroy(surface);
  cairo_surface_destroy(target);

  // Deallocate each triangle and the entire array of triangles
  for (int i = 0; i < num_triangles; i++)
  {
    free(triangles[i]);
  }
  free(triangles);
}

int main(int argc, char *argv[])
{
  // Parse arguments into the global variables
  parse_args(argc, argv);

  // Initialize the global variables
  setup();

  // Main loop
  for (int i = 0; i < iterations; i++)
  {
    // Print out some data every `printSteps` iterations
    if (i % printSteps == 0)
    {
      clock_gettime(CLOCK_REALTIME, &end);
      printf("Iterations: %d, Improvements: %ld, Fitness: %f, Iterations/second: %ld\n", i, timestep, fitness, (long)((double)printSteps / time_diff(start, end)));
      clock_gettime(CLOCK_REALTIME, &start);
    }

    // Advance evolution
    step();
  }

  // Finish up before exiting
  teardown();

  return 0;
}
