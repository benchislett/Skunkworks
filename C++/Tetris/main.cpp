#include <SFML/Graphics.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

// Constant information describing the game

// Blocks are 16 pixels in size, width and height
constexpr uint block_size = 16;

// 22 Rows is standard, but can be extended further if needed
constexpr uint rows = 22;
constexpr uint cols = 10;

// Dimensions of the window
constexpr uint width = 512;
constexpr uint height = 480;

// Location of the upper left corner of the playing field
constexpr uint field_x = 191;
constexpr uint field_y = 63;

// Location of the next piece preview window
constexpr uint next_x = 376;
constexpr uint next_y = 208;

// Size of the preview blocks
constexpr uint mini_block_size = 12;

// Location of the upper left corner of the statistics info window
constexpr uint stats_x = 48;
constexpr uint stats_y = 160;

// Fix framerate to 60 (original tetris runs at 60.09fps)
unsigned long long frame = 1;
const unsigned int framerate = 60;

// Level, line, score and highscore counters
uint level;
uint lines;
uint score;
uint high_score;

// Information needed for scrolling when holding left/right
int horizontalRepeat;
bool horizontalLast;

// Information related to the soft drop mechanic
bool resetDrop;
int dropPoints;

// Track the frequency of each piece for the stats window
uint stats_count[7];

// Tracking the irregular height offset for the stats window pieces
const uint mini_block_offset[7] = {0, 30, 64, 96, 128, 158, 198};

// Point construct, signed to allow coordinates to go negative when hidden above the screen
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

/*
 * Piece construct
 * Stores the coordinate as a `Point` from the upper left corner of the *visible* playing field,
 * Stores the type of the piece as a ushort in the range [0, 8], where 0 is an empty block
 * Stores the rotation as a ushort in the range [0, 3] for each possible configuration in the above table
 */
typedef struct Piece
{
  Point position;
  ushort type;
  ushort rotation = 0;
} Piece;

// Store the current and next piece in memory, updating when a piece is dropped
Piece current;
Piece next;

// Allocate the playing field as a 2D grid of ushort piece types
ushort field[cols][rows];

// Initialize the render window
sf::RenderWindow window(sf::VideoMode(width, height), "Tetris");

// SFML bootstrap

// Background image
sf::Texture background_texture;
sf::Sprite background;

// Store only a single block sprite in memory at a time,
// Pointing to a different tile in the texture map each time
sf::Texture block_textures;
sf::Sprite block_sprite;

// Text objects
sf::Font font;
sf::Text line_text;
sf::Text level_text;
sf::Text score_text;
sf::Text high_score_text;
sf::Text stats_text[7];

// Event to poll from the window
sf::Event e;

// Generate a new piece at random
void new_piece()
{
  current.type = next.type;
  current.position = {3, 0};
  current.rotation = 0;

  next.type = (rand() % 7) + 1;
}

// Reset global variables to the new game state
void reset()
{
  // Clear the field
  for (int i = 0; i < cols; i++)
  {
    for (int k = 0; k < rows; k++)
    {
      field[i][k] = 0;
    }
  }

  // Initialize the position and type of the pieces
  next.type = (rand() % 7) + 1;
  new_piece();

  // Reset the counters
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

// One-time setup for global variables
void setup()
{
  // Seed the random function used to generate the next tile type
  srand(time(NULL));

  // Call the reset function to initialize globals
  reset();

  // Load the highscore from the config file
  std::ifstream high_score_file("data/highscore.cfg", std::ios::in);
  high_score_file >> high_score;
  high_score_file.close();

  // Load the background texture
  background_texture.loadFromFile("data/background.png");
  background.setTexture(background_texture);

  // Load the tilemap
  block_textures.loadFromFile("data/NES_tiles.png");
  block_sprite.setTexture(block_textures);

  // Load the font
  font.loadFromFile("./data/font.ttf");

  // Apply defaults to the text objects

  // Set the font
  line_text.setFont(font);
  level_text.setFont(font);
  score_text.setFont(font);
  high_score_text.setFont(font);

  // Set the font size to 16px
  line_text.setCharacterSize(16);
  level_text.setCharacterSize(16);
  score_text.setCharacterSize(16);
  high_score_text.setCharacterSize(16);

  // Set the position of each piece of text
  line_text.setPosition(304, 46);
  level_text.setPosition(420, 338);
  score_text.setPosition(384, 126);
  high_score_text.setPosition(384, 78);

  // Set each of these for the seven statistics counters
  for (int i = 0; i < 7; i++)
  {
    stats_text[i].setFont(font);
    stats_text[i].setCharacterSize(16);
    stats_text[i].setPosition(stats_x + 4.5 * mini_block_size, stats_y + (2.6 * i + 2.5) * mini_block_size);
    stats_text[i].setFillColor(sf::Color::Red);
  }

  // Fix the framerate
  window.setFramerateLimit(framerate);

  // Do not repeatedly trigger the KeyPressed events when holding a key,
  // We handle that manually
  window.setKeyRepeatEnabled(false);
}

// Since the framerate is fixed, displaying the window steps a single frame forward
void advance_frame()
{
  window.display();
}

// Advance through a number of frames
void advance_frames(int n)
{
  for (int i = 0; i < n; i++)
  {
    advance_frame();
  }
}

// Output the highscore variable to the config file for reuse
void update_high_score()
{
  std::ofstream high_score_file("data/highscore.cfg", std::ios::out);

  high_score = score;
  high_score_file << high_score;

  high_score_file.close();
}

// Pad zeros onto the start of a number until it reaches a fixed number of characters in size
std::string zpad(uint val, uint size)
{
  std::string s = std::to_string(val);
  s.insert(0, size - s.size(), '0');
  return s;
}

// Update the text with the corresponding values
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

// Draw the text to the screen
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

// Reset the background
void draw_background()
{
  window.clear(sf::Color::Blue);
  window.draw(background);
}

// Draw a tile of a given nonzero type to the screen at a coordinate pair
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

// Draw each tile in the field to the screen
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

// Get the x-coordinate of the current piece's i-th block
short get_current_x(short i)
{
  return blocks[current.type - 1][current.rotation][i].x + current.position.x;
}

// Get the y-coordinate of the current piece's i-th block
short get_current_y(short i)
{
  return blocks[current.type - 1][current.rotation][i].y + current.position.y;
}

// Draw the current piece to the screen
void draw_current()
{
  for (short i = 0; i < 4; i++)
  {
    draw_tile(current.type, {get_current_x(i), get_current_y(i)});
  }
}

// Draw the next piece preview to the screen
void draw_next()
{
  for (short i = 0; i < 4; i++)
  {
    // Determine the coordinates of the i-th piece
    int x = next_x + blocks[next.type - 1][0][i].x * block_size;
    int y = next_y + blocks[next.type - 1][0][i].y * block_size;

    // Center the tile in the case of an O or I piece
    if (next.type == 4 || next.type == 7)
    {
      x += block_size / 2;
    }

    // Select the texture tile and move to the coordinates found above
    block_sprite.setTextureRect(sf::IntRect(((next.type - 1) % 3) * block_size, (level % 10) * block_size, block_size, block_size));
    block_sprite.setPosition(x, y);

    // Draw the tile
    window.draw(block_sprite);
  }
}

// Draw the statistics screen mini piece preview to the screen
void draw_stats_pieces()
{
  // For each of the piece types,
  for (short i = 0; i < 7; i++)
  {
    // For each of the tiles in a piece,
    for (short j = 0; j < 4; j++)
    {
      // Determine the coordinates of the piece
      int x = stats_x + blocks[i][0][j].x * mini_block_size;
      int y = stats_y + blocks[i][0][j].y * mini_block_size + mini_block_offset[i];

      // Center the tile in the case of an O or I piece
      if (i == 3 || i == 6)
      {
        x += mini_block_size / 2;
      }

      // Select the texture tile and move to the coordinates found above
      block_sprite.setTextureRect(sf::IntRect((i % 3) * mini_block_size + 4 * block_size, (level % 10) * mini_block_size, mini_block_size, mini_block_size));
      block_sprite.setPosition(x, y);

      // Draw the tile
      window.draw(block_sprite);
    }
  }
}

// Draw the shutters when the game ends
void draw_shutters()
{
  // Select the shutter tile corresponding to the current level
  block_sprite.setTextureRect(sf::IntRect(3 * block_size, (level % 10) * block_size, block_size, block_size));
  for (int i = 2; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      // Draw the row of shutters
      int x = field_x + j * block_size;
      int y = field_y + i * block_size;
      block_sprite.setPosition(x, y);
      window.draw(block_sprite);
    }
    // Pause for 4 frames before drawing the next shutter
    advance_frames(4);
  }
}

// Check if the current piece is in a legal position
bool conflict()
{
  // For each of the tiles in the current piece,
  for (int i = 0; i < 4; i++)
  {
    int x = get_current_x(i);
    int y = get_current_y(i);

    if (x < 0) // Left bound
    {
      return true;
    }
    if (x >= cols) // Right bound
    {
      return true;
    }
    if (y >= rows) // Bottom bound
    {
      return true;
    }
    if (y >= 2 && field[x][y]) // Piece is empty
    {
      return true;
    }
  }
  return false; // No failed checks, position is legal
}

// Run the game over procedure
int game_over()
{
  // Draw the shutters
  draw_shutters();

  // Update the high score if we have done better
  if (score > high_score)
  {
    update_high_score();
  }

  // Reset the globals to a newgame state
  reset();

  // Pause for a second before restarting
  advance_frames(60);
}

// Remove any allocated objects and close the window
void teardown()
{
  window.capture().saveToFile("output/output.png");
  window.close();
}

// Update the score with the number of cleared lines
void update_score(int cleared)
{
  // Update lines and level
  lines += cleared;
  level = lines / 10;

  // Update score: 40, 100, 300, or 1200 points given for 1, 2, 3, or 4 lines cleared, respectively, scaled directly with the level
  score += (cleared == 0 ? 0 : cleared == 1 ? 40 : cleared == 2 ? 100 : cleared == 3 ? 300 : 1200) * (level + 1);
  // Add points for soft drop
  score += dropPoints;
  dropPoints = 0;
}
// Clear any full lines
void check_clear_lines()
{
  std::vector<ushort> to_clear;
  int i, j;
  // For every row on-screen,
  for (i = 2; i < rows; i++)
  {
    // Check if the row contains no empty cells
    bool line_full = true;
    for (j = 0; j < cols; j++)
    {
      if (field[j][i] == 0)
      {
        line_full = false;
        break;
      }
    }

    // Save full line for later
    if (line_full)
    {
      to_clear.push_back(i);
    }
  }

  // Iterate symmetrically from the center outwards
  for (int i = 0; i < cols / 2; i++)
  {
    // Clear from the center of each full row, propagating out from the center
    for (ushort row : to_clear)
    {
      field[cols / 2 + i][row] = 0;
      field[cols / 2 - i - 1][row] = 0;
    }

    // Draw the screen
    draw_background();

    draw_field();

    draw_next();

    draw_stats_pieces();

    draw_text();

    // Pause 4 frames before continuing to clear the lines
    advance_frames(4);
  }

  // Let the rows fall
  for (ushort row : to_clear)
  {
    // Copy from the bottom up, from each cleared row
    for (int i = row; i > 0; i--)
    {
      // And for every cell in that row
      for (int j = 0; j < cols; j++)
      {
        // Copy the above cell downward as we move upward
        field[j][i] = field[j][i - 1];
      }
    }
  }

  // Update the score with the number of cleared lines
  update_score(to_clear.size());
}

// Place the current piece onto the field
void place()
{
  // Increment the counter for current piece type
  stats_count[current.type - 1]++;

  // Copy over the tiles
  for (int i = 0; i < 4; i++)
  {
    field[get_current_x(i)][get_current_y(i)] = current.type;
  }

  // Reset the drop counter so that the down button can't be held between drops
  resetDrop = true;

  // Check if placing this piece has cleared any lines
  check_clear_lines();
}

// Move the piece down one tile
void fall()
{
  // Move the piece
  current.position.y++;

  // If the piece cannot be moved,
  if (conflict())
  {
    // Move it back
    current.position.y--;
    // Place it
    place();
    // Generate a new piece
    new_piece();
  }
}

// Shift the piece left
void left()
{
  current.position.x--;
  if (conflict())
  {
    current.position.x++;
  }
}

// Shift the piece right
void right()
{
  current.position.x++;
  if (conflict())
  {
    current.position.x--;
  }
}

// Shift the piece horizontally depending on which key is pressed
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

// Rotate the piece clockwise
void rotate_cw()
{
  // Increment the rotation and take the modulus 4 to keep it in-bounds
  current.rotation = (current.rotation + 1) % 4;

  if (conflict())
  {
    // If the rotation is not legal, undo
    current.rotation = (current.rotation + 3) % 4;
  }
}

// Rotate the piece counterclockwise
void rotate_ccw()
{
  // Decrement the rotation and take the modulus 4 to keep it in-bounds
  current.rotation = (current.rotation + 3) % 4;

  if (conflict())
  {
    // If the rotation is not legal, undo
    current.rotation = (current.rotation + 1) % 4;
  }
}

// Get the drop speed of the current level, in frames
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

  // Once-time setup
  setup();

  // Game loop
  while (window.isOpen())
  {
    // Draw the playing field
    draw_background();

    draw_field();

    draw_current();

    draw_next();

    draw_stats_pieces();

    draw_text();

    // If the piece's position is illegal, the game is over
    if (conflict())
    {
      game_over();
    }

    // Drop the piece every *drop_rate* frames
    if (frame % drop_rate() == 0)
    {
      fall();
    }
    else
    {
      // If the down arrow key is being held
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down) && !resetDrop)
      {
        // Drop the piece on odd frames
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
        // Logic for holding down horizontal arrow keys
        if (horizontalLast == false)
        {
          horizontalRepeat = 0;
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right) || sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
        {
          // Shift instantly if the key was not previously down
          if (horizontalLast == false)
          {
            shift();
          }

          // Wait 16 frames before shifting first, then 6 frames on subsequent shifts
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

    // Display the window
    advance_frame();

    // Capture any window events
    while (window.pollEvent(e))
    {
      // Finish up when the window is closed
      if (e.type == sf::Event::Closed)
      {
        teardown();
        return 0;
      }

      // Key bindings
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

      // Reset points given for holding the down arrow when the key is released
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

    // Increment the frame count
    frame++;
  }
}
