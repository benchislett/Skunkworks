#include <SFML/Graphics.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

constexpr uint block_size = 16;
constexpr uint rows = 22;
constexpr uint cols = 10;
constexpr uint width = 512;
constexpr uint height = 480;
constexpr uint field_x = 191;
constexpr uint field_y = 63;

unsigned long long frame = 1;
const unsigned int framerate = 60;

uint level = 0;
uint lines = 0;

uint score = 0;
uint high_score = 0;

int horizontalRepeat = 0;
bool horizontalLast = false;

bool resetDrop = false;
int dropPoints = 0;

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

sf::Font font;
sf::Text line_text;
sf::Text level_text;
sf::Text score_text;
sf::Text high_score_text;

sf::Event e;

void setup()
{
  srand(time(NULL));
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

  std::ifstream high_score_file("data/highscore.cfg", std::ios::in);

  high_score_file >> high_score;

  high_score_file.close();

  background_texture.loadFromFile("data/background.png");
  background.setTexture(background_texture);

  block_textures.loadFromFile("data/NES_tiles.png");
  block_sprite.setTexture(block_textures);

  font.loadFromFile("./data/font.ttf");

  line_text.setFont(font);
  level_text.setFont(font);
  score_text.setFont(font);
  high_score_text.setFont(font);

  line_text.setCharacterSize(16);
  level_text.setCharacterSize(16);
  score_text.setCharacterSize(16);
  high_score_text.setCharacterSize(16);

  line_text.setPosition(304, 46);
  level_text.setPosition(420, 338);
  score_text.setPosition(384, 126);
  high_score_text.setPosition(384, 78);

  window.setFramerateLimit(framerate);
  window.setKeyRepeatEnabled(false);
}

void update_high_score()
{
  std::ofstream high_score_file("data/highscore.cfg", std::ios::out);

  high_score = score;
  high_score_file << high_score;

  high_score_file.close();
}

std::string zpad(uint val, uint size)
{
  std::string s = std::to_string(val);
  s.insert(0, size - s.size(), '0');
  return s;
}

void update_text()
{
  line_text.setString(zpad(lines, 3));
  level_text.setString(zpad(level, 2));
  score_text.setString(zpad(score, 6));
  high_score_text.setString(zpad(high_score, 6));
}

void draw_text()
{
  update_text();
  window.draw(line_text);
  window.draw(level_text);
  window.draw(score_text);
  window.draw(high_score_text);
}

void draw_background()
{
  window.clear(sf::Color::Blue);
  window.draw(background);

  draw_text();
}

void draw_tile(ushort type, Point coords)
{
  if (type && coords.y > 1)
  {
    // sf::IntRect(int x, int y, int width, int height)
    block_sprite.setTextureRect(sf::IntRect((type - 1) * block_size, (level % 10) * block_size, block_size, block_size));
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

short get_current_x(short i)
{
  return blocks[current.type - 1][current.rotation][i].x + current.position.x;
}
short get_current_y(short i)
{
  return blocks[current.type - 1][current.rotation][i].y + current.position.y;
}

void draw_current()
{
  for (short i = 0; i < 4; i++)
  {
    draw_tile(current.type, {get_current_x(i), get_current_y(i)});
  }
}

bool conflict()
{
  for (int i = 0; i < 4; i++)
  {
    int x = get_current_x(i);
    int y = get_current_y(i);

    if (x < 0)
    {
      return true;
    }
    if (x >= cols)
    {
      return true;
    }
    if (y >= rows)
    {
      return true;
    }
    if (y >= 2 && field[x][y])
    {
      return true;
    }
  }
  return false;
}

int game_over()
{
  if (score > high_score)
  {
    update_high_score();
  }

  window.capture().saveToFile("output.png");
  window.close();

  return 0;
}

void update_score(int cleared)
{
  lines += cleared;
  level = lines / 10;

  score += (cleared == 0 ? 0 : cleared == 1 ? 40 : cleared == 2 ? 100 : cleared == 3 ? 300 : 1200) * (level + 1);
  score += dropPoints;
  dropPoints = 0;
}

void check_clear_lines()
{
  std::vector<ushort> to_clear;
  int i, j;
  for (i = 2; i < rows; i++)
  {
    bool line_full = true;
    for (j = 0; j < cols; j++)
    {
      if (field[j][i] == 0)
      {
        line_full = false;
        break;
      }
    }
    if (line_full)
    {
      to_clear.push_back(i);
    }
  }

  for (int i = 0; i < cols / 2; i++)
  {
    for (ushort row : to_clear)
    {
      field[cols / 2 + i][row] = 0;
      field[cols / 2 - i - 1][row] = 0;
    }

    draw_background();

    draw_field();

    for (int j = 0; j < 4; j++)
    {
      window.display();
    }
  }

  for (ushort row : to_clear)
  {
    for (int i = row; i > 2; i--)
    {
      for (int j = 0; j < cols; j++)
      {
        field[j][i] = field[j][i - 1];
      }
    }
  }

  update_score(to_clear.size());
}

void place()
{
  for (int i = 0; i < 4; i++)
  {
    field[get_current_x(i)][get_current_y(i)] = current.type;
  }
  resetDrop = true;
  check_clear_lines();
}

void new_piece()
{
  current.type = next.type;
  current.position = {3, 0};
  current.rotation = 0;

  next.type = (rand() % 7) + 1;
}

void fall()
{
  current.position.y++;
  if (conflict())
  {
    current.position.y--;
    place();
    new_piece();
  }
}

void left()
{
  current.position.x--;
  if (conflict())
  {
    current.position.x++;
  }
}

void right()
{
  current.position.x++;
  if (conflict())
  {
    current.position.x--;
  }
}

void shift()
{
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
  {
    right();
  }
  else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
  {
    left();
  }
}

void rotate_cw()
{
  current.rotation = (current.rotation + 1) % 4;
  if (conflict())
  {
    current.rotation = (current.rotation + 3) % 4;
  }
}

void rotate_ccw()
{
  current.rotation = (current.rotation + 3) % 4;
  if (conflict())
  {
    current.rotation = (current.rotation + 1) % 4;
  }
}

int drop_rate()
{
  switch (level)
  {
  case 0:
    return 48;
  case 1:
    return 43;
  case 2:
    return 38;
  case 3:
    return 33;
  case 4:
    return 28;
  case 5:
    return 23;
  case 6:
    return 18;
  case 7:
    return 13;
  case 8:
    return 8;
  case 9:
    return 6;
  case 10:
  case 11:
  case 12:
    return 5;
  case 13:
  case 14:
  case 15:
    return 4;
  case 16:
  case 17:
  case 18:
    return 3;
  case 19:
  case 20:
  case 21:
  case 22:
  case 23:
  case 24:
  case 25:
  case 26:
  case 27:
  case 28:
    return 2;
  default:
    return 1;
  }
}

int main()
{

  setup();

  // game loop
  while (window.isOpen())
  {
    draw_background();

    draw_field();

    draw_current();

    if (conflict())
    {
      std::cout << "Game over!" << std::endl;
      game_over();
    }

    if (frame % drop_rate() == 0)
    {
      fall();
    }
    else
    {
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down) && !resetDrop)
      {
        if (frame % 2)
        {
          fall();
        }
      }
      else
      {
        if (horizontalLast == false)
        {
          horizontalRepeat = 0;
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right) || sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
        {
          if (horizontalLast == false)
          {
            shift();
          }

          if (++horizontalRepeat >= 16)
          {
            horizontalRepeat = 10;
            shift();
          }
          horizontalLast = true;
        }
        else
        {
          horizontalLast = false;
        }
      }
    }

    window.display();

    while (window.pollEvent(e))
    {
      if (e.type == sf::Event::Closed)
      {
        game_over();
      }

      if (e.type == sf::Event::KeyPressed)
      {
        switch (e.key.code)
        {
        case sf::Keyboard::Z:
          rotate_cw();
          break;
        case sf::Keyboard::X:
          rotate_ccw();
          break;
        case sf::Keyboard::Down:
          resetDrop = false;
          break;
        }
      }
    }

    frame++;
  }
}
