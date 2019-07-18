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

constexpr uint next_x = 376;
constexpr uint next_y = 208;

constexpr uint mini_block_size = 12;
constexpr uint stats_x = 48;
constexpr uint stats_y = 160;

unsigned long long frame = 1;
const unsigned int framerate = 60;

uint level;
uint lines;

uint score;
uint high_score;

int horizontalRepeat;
bool horizontalLast;

bool resetDrop;
int dropPoints;

uint stats_count[7];
const uint mini_block_offset[7] = {0, 30, 64, 96, 128, 158, 198};

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
    {{{1, 2}, {2, 2}, {2, 3}, {3, 2}}, // T Piece
     {{2, 1}, {2, 2}, {2, 3}, {3, 2}},
     {{1, 2}, {2, 2}, {2, 1}, {3, 2}},
     {{1, 2}, {2, 2}, {2, 3}, {2, 1}}},
    {{{1, 2}, {2, 2}, {3, 2}, {3, 3}}, // J Piece
     {{2, 3}, {2, 2}, {2, 1}, {3, 1}},
     {{1, 1}, {1, 2}, {2, 2}, {3, 2}},
     {{1, 3}, {2, 3}, {2, 2}, {2, 1}}},
    {{{1, 2}, {2, 2}, {2, 3}, {3, 3}}, // Z Piece
     {{2, 3}, {2, 2}, {3, 2}, {3, 1}},
     {{1, 2}, {2, 2}, {2, 3}, {3, 3}},
     {{2, 3}, {2, 2}, {3, 2}, {3, 1}}},
    {{{1, 3}, {1, 2}, {2, 2}, {2, 3}}, // O Piece
     {{1, 3}, {1, 2}, {2, 2}, {2, 3}},
     {{1, 3}, {1, 2}, {2, 2}, {2, 3}},
     {{1, 3}, {1, 2}, {2, 2}, {2, 3}}},
    {{{1, 3}, {2, 3}, {2, 2}, {3, 2}}, // S Piece
     {{2, 1}, {2, 2}, {3, 2}, {3, 3}},
     {{1, 3}, {2, 3}, {2, 2}, {3, 2}},
     {{2, 1}, {2, 2}, {3, 2}, {3, 3}}},
    {{{1, 3}, {1, 2}, {2, 2}, {3, 2}}, // L Piece
     {{2, 1}, {2, 2}, {2, 3}, {3, 3}},
     {{1, 2}, {2, 2}, {3, 2}, {3, 1}},
     {{1, 1}, {2, 1}, {2, 2}, {2, 3}}},
    {{{0, 2}, {1, 2}, {2, 2}, {3, 2}}, // I Piece
     {{2, 0}, {2, 1}, {2, 2}, {2, 3}},
     {{0, 2}, {1, 2}, {2, 2}, {3, 2}},
     {{2, 0}, {2, 1}, {2, 2}, {2, 3}}}};

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
sf::Text stats_text[7];

sf::Event e;

void reset()
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

  level = 0;
  lines = 0;

  score = 0;

  horizontalRepeat = 0;
  horizontalLast = false;

  resetDrop = false;
  dropPoints = 0;

  for (int i = 0; i < 7; i++)
  {
    stats_count[i] = 0;
  }
}

void setup()
{
  srand(time(NULL));

  reset();

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

  for (int i = 0; i < 7; i++)
  {
    stats_text[i].setFont(font);
    stats_text[i].setCharacterSize(16);
    stats_text[i].setPosition(stats_x + 4.5 * mini_block_size, stats_y + (2.6 * i + 2.5) * mini_block_size);
    stats_text[i].setFillColor(sf::Color::Red);
  }

  window.setFramerateLimit(framerate);
  window.setKeyRepeatEnabled(false);
}

void advance_frame()
{
  window.display();
}

void advance_frames(int n)
{
  for (int i = 0; i < n; i++)
  {
    advance_frame();
  }
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

  for (int i = 0; i < 7; i++)
  {
    stats_text[i].setString(zpad(stats_count[i], 3));
  }
}

void draw_text()
{
  update_text();
  window.draw(line_text);
  window.draw(level_text);
  window.draw(score_text);
  window.draw(high_score_text);
  for (int i = 0; i < 7; i++)
  {
    window.draw(stats_text[i]);
  }
}

void draw_background()
{
  window.clear(sf::Color::Blue);
  window.draw(background);
}

void draw_tile(ushort type, Point coords)
{
  if (type && coords.y > 1)
  {
    // sf::IntRect(int x, int y, int width, int height)
    block_sprite.setTextureRect(sf::IntRect(((type - 1) % 3) * block_size, (level % 10) * block_size, block_size, block_size));
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

void draw_next()
{
  for (short i = 0; i < 4; i++)
  {
    int x = next_x + blocks[next.type - 1][0][i].x * block_size;
    int y = next_y + blocks[next.type - 1][0][i].y * block_size;

    if (next.type == 4 || next.type == 7)
    {
      x += block_size / 2;
    }

    block_sprite.setTextureRect(sf::IntRect(((next.type - 1) % 3) * block_size, (level % 10) * block_size, block_size, block_size));
    block_sprite.setPosition(x, y);

    window.draw(block_sprite);
  }
}

void draw_stats_pieces()
{
  for (short i = 0; i < 7; i++)
  {
    for (short j = 0; j < 4; j++)
    {
      int x = stats_x + blocks[i][0][j].x * mini_block_size;
      int y = stats_y + blocks[i][0][j].y * mini_block_size + mini_block_offset[i];

      if (i == 3 || i == 6)
      {
        x += mini_block_size / 2;
      }

      block_sprite.setTextureRect(sf::IntRect((i % 3) * mini_block_size + 4 * block_size, (level % 10) * mini_block_size, mini_block_size, mini_block_size));
      block_sprite.setPosition(x, y);

      window.draw(block_sprite);
    }
  }
}

void draw_shutters()
{
  block_sprite.setTextureRect(sf::IntRect(3 * block_size, (level % 10) * block_size, block_size, block_size));
  for (int i = 2; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      int x = field_x + j * block_size;
      int y = field_y + i * block_size;
      block_sprite.setPosition(x, y);
      window.draw(block_sprite);
    }
    advance_frames(4);
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
  draw_shutters();

  if (score > high_score)
  {
    update_high_score();
  }

  reset();

  advance_frames(60);
}

void teardown()
{
  window.capture().saveToFile("output/output.png");
  window.close();
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

    draw_next();

    draw_stats_pieces();

    draw_text();

    advance_frames(4);
  }

  for (ushort row : to_clear)
  {
    for (int i = row; i > 0; i--)
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
  stats_count[current.type - 1]++;
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

    draw_next();

    draw_stats_pieces();

    draw_text();

    if (conflict())
    {
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
          // Mimic a bug in classic tetris that limits and resets dropPoints
          if (++dropPoints > 15)
          {
            dropPoints = 10;
          }
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

    advance_frame();

    while (window.pollEvent(e))
    {
      if (e.type == sf::Event::Closed)
      {
        teardown();
        return 0;
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
          dropPoints = 0;
          break;
        case sf::Keyboard::Q:
          teardown();
          return 0;
        }
      }

      if (e.type == sf::Event::KeyReleased)
      {
        switch (e.key.code)
        {
        case sf::Keyboard::Down:
          dropPoints = 0;
          break;
        }
      }
    }

    frame++;
  }
}
