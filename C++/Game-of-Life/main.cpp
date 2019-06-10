#include <SFML/Graphics.hpp>
#include <iostream>
#include <cmath>

int width = 1000;
int height = 600;
constexpr int rows = 60;
constexpr int cols = 100;

int cell_width = width / cols;
int cell_height = height / rows;

bool cells_a[cols][rows];
bool cells_b[cols][rows];

int timestep;

sf::RenderWindow window(sf::VideoMode(width, height), "SFML");

sf::RectangleShape dead;
sf::RectangleShape alive;

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

  dead.setSize(sf::Vector2f(cell_width, cell_height));
  dead.setOutlineColor(sf::Color::Black);
  dead.setOutlineThickness(1);
  dead.setPosition(0, 0);

  alive.setSize(sf::Vector2f(cell_width, cell_height));
  alive.setOutlineColor(sf::Color::Black);
  alive.setOutlineThickness(1);
  alive.setPosition(0, 0);

  dead.setFillColor(sf::Color::White);
  alive.setFillColor(sf::Color::Black);
}

void step()
{
  bool(*cells)[rows] = (timestep % 2) ? cells_a : cells_b;
  timestep++;
  bool(*other)[rows] = (timestep % 2) ? cells_a : cells_b;

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

void draw()
{
  window.clear();

  bool(*cells)[rows] = (timestep % 2) ? cells_a : cells_b;

  for (int i = 0; i < cols; i++)
  {
    for (int j = 0; j < rows; j++)
    {
      sf::RectangleShape block = cells[i][j] ? alive : dead;
      block.setPosition(i * cell_width, j * cell_height);
      window.draw(block);
    }
  }

  window.display();
}

int main()
{
  reset();
  while (window.isOpen())
  {
    sf::Vector2u size = window.getSize();
    width = size.x;
    height = size.y;

    draw();

    sf::Event event;
    while (window.pollEvent(event))
    {
      if (event.type == sf::Event::Closed)
      {
        window.capture().saveToFile("output/output.png");
        window.close();

        return 0;
      }
      else if (event.type == sf::Event::MouseButtonPressed)
      {
        sf::Vector2i MousePos = sf::Mouse::getPosition(window);

        int x = MousePos.x;
        int y = MousePos.y;
        if (0 < x && x < width && 0 < y && y < height)
        {
          int col = floor(x / (float)width * cols);
          int row = floor(y / (float)height * rows);
          ((timestep % 2) ? cells_a : cells_b)[col][row] = !((timestep % 2) ? cells_a : cells_b)[col][row];
        }
      }
      else if (event.type == sf::Event::KeyPressed)
      {
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
        {
          step();
        }
      }
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
    {
      step();
    }
  }
  return 0;
}
