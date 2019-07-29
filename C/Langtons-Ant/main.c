// Enable some functionality for cairo
#define _POSIX_C_SOURCE 200809L

#include <cairo.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int width = 256;
int height = 256;

// Arguments with defaults
unsigned int iterations = 1000;

// Set the global timestep counter
long int timestep = 1;

// Length of progress bar
int barlen = 12;

// Surface for the main image, and the cairo instance
cairo_surface_t *surface;
cairo_t *cr;

struct timespec start;
struct timespec end;

// Parse the command arguments into the global variables
void parse_args(int argc, char *argv[])
{
  int opt;
  while ((opt = getopt(argc, argv, "i:")) != -1)
  {
    switch (opt)
    {
    case 'i':
      // Number of iterations to run the simulation
      iterations = atoi(optarg);
      break;
    default:
      // Print a help script if invalid arguments are entered
      fprintf(stderr, "Usage: %s [-i iterations] \n", argv[0]);
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
}

// Wipe the canvas
void set_background_colour()
{
  cairo_set_source_rgb(cr, 0, 0, 0);
  cairo_rectangle(cr, 0, 0, width, height);
  cairo_fill(cr);
}

// Fix the background colour
void draw()
{
  set_background_colour();
}

void show_progress(double i, double imax) {
  printf("[");
  int progress = barlen * (i / imax);
  for (int j = 0; j < barlen; j++) {
    printf(j <= progress ? "." : " ");
  }
  printf("]\r");
  fflush(stdout);
}

// Advance the simulation
void step()
{
  draw();
}

// Destroy, deallocate, and conclude operations
void teardown()
{
  // Destroy cairo objects
  cairo_destroy(cr);
  cairo_surface_write_to_png(surface, "output/output.png");
  cairo_surface_destroy(surface);

  printf("\n");
}

int main(int argc, char *argv[])
{
  // Parse arguments into the global variables
  parse_args(argc, argv);

  // Initialize the program
  setup();

  // Main loop
  for (int i = 0; i < iterations; i++)
  {
    // Advance evolution
    step();

    // Display progress bar
    show_progress((double)i, (double)iterations);
  }

  // Finish up before exiting
  teardown();

  return 0;
}
