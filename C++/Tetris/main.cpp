#include <SFML/Graphics.hpp>
#include <iostream>

constexpr int block_size = 16;
constexpr int rows = 20;
constexpr int cols = 10;
constexpr int width = 512;
constexpr int height = 480;
constexpr int field_x = 191;
constexpr int field_y = 95;

unsigned long long frame = 0;
const unsigned int framerate = 60;

typedef struct Point
{
  short x;
  short y;
} Point;

/*
 * Defines a 4x4 grid within which each piece is located
 * Each block is defined as a set of 4 rotations,
 * each of which has 4 coordinates in the grid, with the upper leftmost square being {0, 0}
 * This grid initially spans the square with corners coordinates (3, -2) through (6, 1) inclusive when the piece spawns 
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
  Point position;
  ushort type;
  ushort rotation = 0;
} Piece;

Piece current;
Piece next;

ushort field[cols][rows];

sf::RenderWindow window(sf::VideoMode(width, height), "Tetris");

sf::Texture background_texture;
sf::Sprite background;

sf::Texture block_textures;
sf::Sprite block_sprite;

sf::Event e;

void setup()
{
  for (int i = 0; i < cols; i++)
  {
    for (int k = 0; k < rows; k++)
    {
      field[i][k] = 0;
    }
  }

  current.position = {3, 0};
  current.type = (rand() % 7) + 1;

  next.type = (rand() % 7) + 1;

  background_texture.loadFromFile("data/background.png");
  background.setTexture(background_texture);

  block_textures.loadFromFile("data/NES_tiles.png");
  block_sprite.setTexture(block_textures);

  window.setFramerateLimit(framerate);
}

void draw_background()
{
  window.clear(sf::Color::Blue);
  window.draw(background);
}

void draw_tile(ushort type, Point coords)
{
  if (type)
  {
    // sf::IntRect(int x, int y, int width, int height)
    block_sprite.setTextureRect(sf::IntRect((type - 1) * block_size, 0, block_size, block_size));
    block_sprite.setPosition(field_x + coords.x * block_size, field_y + coords.y * block_size);

    window.draw(block_sprite);
  }
}

void draw_field()
{
  for (short i = 0; i < cols; i++)
  {
    for (short k = 0; k < rows; k++)
    {
      draw_tile(field[i][k], {i, k});
    }
  }
}

bool conflict()
{
  for (int i = 0; i < 4; i++)
  {
    int x = blocks[current.type - 1][current.rotation][i].x + current.position.x;
    int y = blocks[current.type - 1][current.rotation][i].y + current.position.y;

    if (x < 0 || x >= cols || y >= rows || field[x][y])
    {
      return false;
    }
  }
  return true;
}

int game_over()
{
  window.capture().saveToFile("output.png");
  window.close();

  return 0;
}

int main()
{

  setup();

  // game loop
  while (window.isOpen())
  {
    draw_background();

    draw_field();

    if (conflict())
    {
      game_over();
    }

    window.display();

    while (window.pollEvent(e))
    {
      if (e.type != sf::Event::Closed)
      {
        game_over();
      }
    }

    frame++;
  }
}
