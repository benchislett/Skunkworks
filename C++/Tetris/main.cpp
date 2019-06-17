#include <SFML/Graphics.hpp>

#define block_size 16
#define rows 20
#define cols 10
#define width 512
#define height 480
#define field_x 191
#define field_y 95

typedef ushort coord;

typedef struct Point
{
  coord x;
  coord y;
} Point;

/*
 * Defines a 4x4 grid within which each piece is located
 * Each block is defined as a set of 4 rotations,
 * each of which has 4 coordinates in the grid, with the upper leftmost square being {0, 0}
 * This grid initially spans the square with corners coordinates (3, -2) through (6, 1) inclusive when the piece spawnws 
 */
const Point blocks[7][4][4] = {
    {{{0, 2}, {1, 2}, {2, 2}, {3, 2}}, // I Piece
     {{2, 0}, {2, 1}, {2, 2}, {2, 3}},
     {{0, 2}, {1, 2}, {2, 2}, {3, 2}},
     {{2, 0}, {2, 1}, {2, 2}, {2, 3}}},
    {{{1, 2}, {2, 2}, {2, 3}, {3, 3}}, // Z Piece
     {{2, 3}, {2, 2}, {3, 2}, {3, 1}},
     {{1, 2}, {2, 2}, {2, 3}, {3, 3}},
     {{2, 3}, {2, 2}, {3, 2}, {3, 1}}},
    {{{1, 3}, {2, 3}, {2, 2}, {3, 2}}, // S Piece
     {{2, 1}, {2, 2}, {3, 2}, {3, 3}},
     {{1, 3}, {2, 3}, {2, 2}, {3, 2}},
     {{2, 1}, {2, 2}, {3, 2}, {3, 3}}},
    {{{1, 2}, {2, 2}, {2, 3}, {3, 2}}, // T Piece
     {{2, 1}, {2, 2}, {2, 3}, {3, 2}},
     {{1, 2}, {2, 2}, {2, 1}, {3, 2}},
     {{1, 2}, {2, 2}, {2, 3}, {2, 1}}},
    {{{1, 3}, {1, 2}, {2, 2}, {3, 2}}, // L Piece
     {{2, 1}, {2, 2}, {2, 3}, {3, 3}},
     {{1, 2}, {2, 2}, {3, 2}, {3, 1}},
     {{1, 1}, {2, 1}, {2, 2}, {2, 3}}},
    {{{1, 2}, {2, 2}, {3, 2}, {3, 3}}, // J Piece
     {{2, 3}, {2, 2}, {2, 1}, {3, 1}},
     {{1, 1}, {1, 2}, {2, 2}, {3, 2}},
     {{1, 3}, {2, 3}, {2, 2}, {2, 1}}},
    {{{1, 3}, {1, 2}, {2, 2}, {2, 3}}, // O Piece
     {{1, 3}, {1, 2}, {2, 2}, {2, 3}},
     {{1, 3}, {1, 2}, {2, 2}, {2, 3}},
     {{1, 3}, {1, 2}, {2, 2}, {2, 3}}}};

typedef struct Piece
{
  Point coords;
  ushort rotation = 0;
} Piece;

void setup()
{
}

void main()
{
}
