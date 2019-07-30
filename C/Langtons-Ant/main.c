// Enable some functionality for cairo
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

// Arguments with defaults
unsigned long max_iterations = 250000;
int width = 256;
int height = 256;
char fname[64] = "output/output.bmp";
char pattern[16] = "LRRRRRLLR";

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
  unsigned char r;
  unsigned char g;
  unsigned char b;
} Colour;

// Number of states
int states;

// Parse the command arguments into the global variables
void parse_args(int argc, char *argv[])
{
  int opt;
  while ((opt = getopt(argc, argv, "i:x:y:o:p:")) != -1)
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
    case 'o':
      // Output file name (relative to executable)
      strcpy(fname, optarg);
      break;
    case 'p':
      // Decision pattern
      strcpy(pattern, optarg);
      break;
    default:
      // Print a help script if invalid arguments are entered
      fprintf(stderr, "Usage: %s [-i max_iters] [-x width] [-y height] [-o output_file] \n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

// Initialize the global variables
void setup()
{
  // Declare the number of states to the field
  states = strlen(pattern);

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
      return (Colour){255, 255, 255};
    case 1:
      return (Colour){93, 39, 93};
    case 2:
      return (Colour){177, 62, 83};
    case 3:
      return (Colour){239, 125, 87};
    case 4:
      return (Colour){255, 205, 117};
    case 5:
      return (Colour){167, 240, 112};
    case 6:
      return (Colour){56, 183, 100};
    case 7:
      return (Colour){37, 113, 121};
    case 8:
      return (Colour){41, 54, 111};
    case 9:
      return (Colour){59, 93, 201};
    case 10:
      return (Colour){65, 166, 249};
    case 11:
      return (Colour){115, 239, 247};
    case 12:
      return (Colour){148, 176, 194};
    case 13:
      return (Colour){86, 108, 134};
    case 14:
      return (Colour){51, 60, 87};
    case 15:
      return (Colour){26, 28, 44};
  }
}

// Output the rendered image in bitmap format
void draw()
{
  // Header bytes
  static unsigned char header[54] = {66,77,0,0,0,0,0,0,0,0,54,0,0,0,40,0,0,0,0,0,0,0,0,0,0,0,1,0,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  
  // Target file
  FILE *fp = fopen(fname, "w");
  
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
      data[3 * (j * width + i)] = c.b;
      data[3 * (j * width + i) + 1] = c.g;
      data[3 * (j * width + i) + 2] = c.r;
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
  char action = pattern[field[x][y]];
  switch (action) {
    case 'R':
      rotate_n(1);
      break;
    case 'L':
      rotate_n(-1);
      break;
    case 'U':
      rotate_n(2);
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
