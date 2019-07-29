// Enable some functionality for cairo
#define _POSIX_C_SOURCE 200809L

#include <cairo.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define width 256
#define height 256

// Arguments with defaults
unsigned int max_iterations = 20000;

// Set the global timestep counter
long timestep = 1;

// Length of progress bar
int barlen = 72;

// Create the game state
int field[width][height];
short direction; // { 0: N, 1: E, 2: S, 3: W }
long x;
long y;

// Surface for the main image, and the cairo instance
cairo_surface_t *surface;
cairo_t *cr;

// Structure for colours
typedef struct {
  double r;
  double g;
  double b;
} Colour;

// Parse the command arguments into the global variables
void parse_args(int argc, char *argv[])
{
  int opt;
  while ((opt = getopt(argc, argv, "i:")) != -1)
  {
    switch (opt)
    {
    case 'i':
      // Number of iterations to run the simulation before exiting
      max_iterations = atoi(optarg);
      break;
    default:
      // Print a help script if invalid arguments are entered
      fprintf(stderr, "Usage: %s [-i max_iters] \n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

// Initialize the global variables
void setup()
{
  // Create the context for the main image
  surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, width, height);
  cr = cairo_create(surface);

  // Clear the field
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      field[i][j] = 0;
    }
  }

  // Set the starting position and orientation
  x = width / 2;
  y = height / 2;
  direction = 0;
}

// Destroy, deallocate, and conclude operations
void teardown()
{
  // Destroy cairo objects
  cairo_destroy(cr);
  cairo_surface_write_to_png(surface, "output/output.png");
  cairo_surface_destroy(surface);

  printf("\n");
  exit(0);
}

// Wipe the canvas
void set_background_colour()
{
  cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
  cairo_rectangle(cr, 0, 0, width, height);
  cairo_fill(cr);
}

// Draw a single pixel
void draw_pixel(long x, long y, Colour c) {
  // Don't waste time drawing pixels that have the same colour as the background
  if (c.r != 0.0 || c.g != 0.0 || c.b != 0.0) {
    cairo_set_source_rgb(cr, c.r, c.g, c.b);
    cairo_rectangle(cr, x, y, 1, 1);
    cairo_fill(cr);
  }
}

// Get the Colour of a given pixel
Colour get_colour(int x, int y) {
  switch (field[x][y]) {
    case 0:
      return (Colour){1.0, 1.0, 1.0};
    case 1:
      return (Colour){0.0, 0.0, 0.0};
  }
}

// Fix the background colour
void draw()
{
  set_background_colour();
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      draw_pixel(i, j, get_colour(i, j));
    }
  }
}

// Show a progress bar towards the iteration limit
void show_progress(double i, double imax) {
  printf("[");
  int progress = barlen * (i / imax);
  for (int j = 0; j < barlen; j++) {
    printf(j <= progress ? "." : " ");
  }
  printf("]\r");
  fflush(stdout);
}

// Rotate the ant n*90 degrees clockwise 
void rotate_n(int n) {
  while (n < 0) {
    n += 4;
  }

  direction = (direction + n) % 4;
}

// Rotate the ant based on the colour of the tile
void rotate() {
  switch (field[x][y]) {
    case 0:
      rotate_n(1);
      break;
    case 1:
      rotate_n(-1);
      break;
  }
}

// Modify the state of the tile the ant is currently standing on
void flip() {
  field[x][y] = (field[x][y] + 1) % 2;
}

// Advance the position of the ant a single unit in its current direction
void walk() {
  switch (direction) {
    case 0:
      y--;
      break;
    case 1:
      x++;
      break;
    case 2:
      y++;
      break;
    case 3:
      x--;
      break;
  }
}

void check() {
  if (x == 0 || x + 1 == width || y == 0 || y + 1 == height) {
    teardown();
  }
}

// Advance the simulation
void step()
{
  rotate();
  flip();
  walk();
  check();
  draw();
}

int main(int argc, char *argv[])
{
  // Parse arguments into the global variables
  parse_args(argc, argv);

  // Initialize the program
  setup();

  // Main loop
  for (int i = 0; i < max_iterations; i++)
  {
    // Advance evolution
    step();

    // Display progress bar
    show_progress((double)i, (double)max_iterations);
  }

  // Finish up and exit
  teardown();

  return 0;
}
