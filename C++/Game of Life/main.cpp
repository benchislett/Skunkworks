#include <SFML/Graphics.hpp>
#include <iostream>

constexpr int width = 200;
constexpr int height = 200;
constexpr int rows = 20;
constexpr int cols = 20;

bool cells_a[cols][rows];
bool cells_b[cols][rows];

int timestep;

sf::RenderWindow window(sf::VideoMode(width, height), "SFML");

bool within_bounds(int x, int y)
{
  return (x < width) && (0 <= x) && (y < height) && (0 <= y);
}

bool next_state(bool alive, int neighbours)
{
  if (alive && (neighbours == 2 || neighbours == 3))
  {
    return true;
  }
  else if (!alive && neighbours == 3)
  {
    return true;
  }
  else
  {
    return false;
  }
}

void reset()
{
  timestep = 0;
  for (int i = 0; i < cols; i++)
  {
    for (int j = 0; j < rows; j++)
    {
      cells_a[i][j] = false;
      cells_b[i][j] = false;
    }
  }
}

void step()
{
  bool(*cells)[20] = (++timestep % 2) ? cells_a : cells_b;
  bool(*other)[20] = (timestep % 2) ? cells_a : cells_b;

  int sides[8][2] = {
      {-1, -1},
      {-1, 0},
      {-1, 1},
      {0, -1},
      {0, 1},
      {1, -1},
      {1, 0},
      {1, 1}};

  int neighbours;
  for (int i = 1; i < cols - 1; i++)
  {
    for (int j = 1; j < rows - 1; j++)
    {
      neighbours = 0;
      for (int k = 0; k < 8; k++)
      {
        if (cells[i + sides[k][0]][j + sides[k][1]])
        {
          neighbours++;
        }
      }

      other[i][j] = next_state(cells[i][j], neighbours);
    }
  }
}

int main()
{
  reset();
  window.close();
  return 0;
}
