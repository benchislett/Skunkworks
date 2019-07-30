// Enable some functionality for cairo
#define _POSIX_C_SOURCE 200809L

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Arguments with defaults
unsigned long max_iterations = 250000;
int width = 256;
int height = 256;

// Set the global timestep counter
long timestep = 1;

// Length of progress bar
int barlen = 64;

// Create the game state
int **field;
short direction; // { 0: N, 1: E, 2: S, 3: W }
long x;
long y;

// Structure for colours
typedef struct {
  double r;
  double g;
  double b;
} Colour;

// Number of states
int states = 4;

// Parse the command arguments into the global variables
void parse_args(int argc, char *argv[])
{
  int opt;
  while ((opt = getopt(argc, argv, "i:x:y:")) != -1)
  {
    switch (opt)
    {
    case 'i':
      // Number of iterations to run the simulation before exiting
      max_iterations = atoi(optarg);
      break;
    case 'x':
      // Width
      width = atoi(optarg);
      break;
    case 'y':
      // Height
      height = atoi(optarg);
      break;
    default:
      // Print a help script if invalid arguments are entered
      fprintf(stderr, "Usage: %s [-i max_iters] [-x width] [-y height] \n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

// Initialize the global variables
void setup()
{
  // Allocate memory for the field
  field = (int **)malloc(width * sizeof(int *));
  for (int i = 0; i < width; i++) {
    field[i] = (int *)calloc(height, sizeof(int));
  }

  // Set the starting position and orientation
  x = width / 2;
  y = height / 2;
  direction = 0;
}

// Get the Colour of a given pixel
Colour get_colour(int x, int y) {
  switch (field[x][y]) {
    case 0:
      return (Colour){1.0, 1.0, 1.0};
    case 1:
      return (Colour){0.5, 0.0, 1.0};
    case 2:
      return (Colour){1.0, 0.5, 0.0};
    case 3:
      return (Colour){0.0, 1.0, 0.5};
  }
}

// Output the rendered image in bitmap format
void draw()
{
  // Header bytes
  static unsigned char header[54] = {66,77,0,0,0,0,0,0,0,0,54,0,0,0,40,0,0,0,0,0,0,0,0,0,0,0,1,0,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  
  // Target file
  FILE *fp = fopen("output/output.bmp", "w");
  
  // Update the file size section of the header
  unsigned int* header_file_size = (unsigned int*) &header[2];
	*header_file_size = 54 + (3 * width + (4 - ((3 * width) % 4)) % 4)*height;
  
  // Update the width and height section of the header
	unsigned int* header_width = (unsigned int*) &header[18];    
	*header_width = width;
	unsigned int* header_height = (unsigned int*) &header[22];    
	*header_height = height;

  // Write the header
  fwrite(header, 54, 1, fp);

  // Fill a bytemap with the field data
  unsigned char *data = (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      Colour c = get_colour(i, j);
      data[3 * (j * width + i)] = (unsigned char)(c.b * 255.0);
      data[3 * (j * width + i) + 1] = (unsigned char)(c.g * 255.0);
      data[3 * (j * width + i) + 2] = (unsigned char)(c.r * 255.0);
    }
  }

  // Write the data
  fwrite(data, 3*width*height, 1, fp);

  // Close the file
  fclose(fp);
}

// Destroy, deallocate, and conclude operations
void teardown()
{
  // Output the field to an image
  draw();

  // Deallocate the field
  for (int i = 0; i < width; i++) {
    free(field[i]);
  }
  free(field);

  printf("\n");
  exit(0);
}

// Show a progress bar towards the iteration limit
void show_progress(double i, double imax) {
  printf("[");
  int progress = barlen * (i / imax);
  for (int j = 0; j < barlen; j++) {
    printf(j <= progress ? "." : " ");
  }
  printf("] Iteration: %ld\r", (long)i);
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
    case 2:
      rotate_n(-1);
      break;
    case 3:
      rotate_n(1);
      break;
  }
}

// Modify the state of the tile the ant is currently standing on
void flip() {
  field[x][y] = (field[x][y] + 1) % states;
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
