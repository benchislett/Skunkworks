#include <SFML/Graphics.hpp>
#include <iostream>
#include <cmath>

// Set the initial window size
// Width and height are dynamically updated when the screen is resized
int width = 1000;
int height = 600;
constexpr int rows = 60;
constexpr int cols = 100;

// Initialize the cell size, dependent on width and height
int cell_width = width / cols;
int cell_height = height / rows;

// Initialize two cell grids, using one to display and the other as a buffer
// Alternate which grid is drawn each timestep to avoid having to copy both ways
bool cells[2][cols][rows];

int timestep;
const int framerate = 60;

// Keep track of which type was initially pressed when "painting" tiles with the right mouse button
bool paint_type = true;

// SFML boilerplate
sf::RenderWindow window(sf::VideoMode(width, height), "SFML");

sf::RectangleShape dead;
sf::RectangleShape alive;

// Check if a pixel coordinate is on-screen
bool within_bounds(int x, int y)
{
  return (x < width) && (0 <= x) && (y < height) && (0 <= y);
}

// Determine the new state of a cell given its current state and number of neighbours
bool next_state(bool alive, int neighbours)
{
  if (alive && (neighbours == 2 || neighbours == 3))
  {
    // Stay alive if neighbour count is optimal
    return true;
  }
  else if (!alive && neighbours == 3)
  {
    // Return to life if neighbouring region is populous
    return true;
  }
  else
  {
    // Cell becomes/stays dead since surroundings are suboptimally populated
    return false;
  }
}

// Clears the entire cell field
void clear()
{
  for (int i = 0; i < cols; i++)
  {
    for (int j = 0; j < rows; j++)
    {
      cells[0][i][j] = false;
      cells[1][i][j] = false;
    }
  }
}

// Reset globals, to initialize a new game
void reset()
{
  timestep = 0;

  // Clear the cells
  clear();

  // Setup the black and white tile textures

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

  // Set the framerate
  window.setFramerateLimit(framerate);

  // Do not repeatedly trigger the KeyPressed events when holding a key,
  // We handle that manually
  window.setKeyRepeatEnabled(false);
}

// Advance forward in time to the next state
void step()
{
  // Advance timestep and select which grid to use
  int parity = timestep++ % 2;

  int neighbours;
  // For every column,
  for (int i = 1; i < cols - 1; i++)
  {
    // For every row,
    for (int j = 1; j < rows - 1; j++)
    {
      // Set the cell's living neighbour count to 0
      neighbours = 0;

      // Check left, center, and right
      for (int x = -1; x < 2; x++)
      {
        // Check top, center, and bottom
        for (int y = -1; y < 2; y++)
        {
          // Ensure we aren't counting the cell itself
          if ((x != 0 || y != 0) && cells[parity][i + x][j + y])
          {
            // If the neighbour cell is alive, increment the count
            neighbours++;
          }
        }
      }
      // Save the next state of the current cell in the other side of the grid mapping
      cells[1 - parity][i][j] = next_state(cells[parity][i][j], neighbours);
    }
  }
}

// Draw the grid to the screen
void draw()
{
  // Clear the window
  window.clear();

  // For every column,
  for (int i = 0; i < cols; i++)
  {
    // For every row,
    for (int j = 0; j < rows; j++)
    {
      // Select the living or dead cell sprite based on the state of the current cell
      sf::RectangleShape block = cells[timestep % 2][i][j] ? alive : dead;

      // Set the coordinates of the cell and draw it
      block.setPosition(i * cell_width, j * cell_height);
      window.draw(block);
    }
  }

  // Display the screen, advancing a single frame
  window.display();
}

// Remove any allocated objects and close the window
int teardown()
{
  // Safe the current window to the output file
  window.capture().saveToFile("output/output.png");
  window.close();

  return 0;
}

int main()
{
  // Initialize the global environment
  reset();

  // Main loop
  while (window.isOpen())
  {
    // Update the size of the window to ensure that mouse position checks are valid
    sf::Vector2u size = window.getSize();
    width = size.x;
    height = size.y;

    // Draw the current state
    draw();

    // Check for window events
    sf::Event event;
    while (window.pollEvent(event))
    {
      if (event.type == sf::Event::Closed)
      {
        // If the window closes, deallocate and terminate
        return teardown();
      }
      else if (event.type == sf::Event::MouseButtonPressed)
      {
        // On mouse click, identify the cell that was clicked and flip its state

        // Find the pixel location of the click event
        sf::Vector2i MousePos = sf::Mouse::getPosition(window);

        // Extract the coordinates of the event
        int x = MousePos.x;
        int y = MousePos.y;

        // If the event is valid,
        if (within_bounds(x, y))
        {
          // Compute the coordinates of the cell which was clicked
          int col = floor(x / (float)width * cols);
          int row = floor(y / (float)height * rows);

          // Flip the state of the selected cell
          cells[timestep % 2][col][row] = !cells[timestep % 2][col][row];

          if (event.mouseButton.button == sf::Mouse::Right)
          {
            // Update the "painting" cell type when right mouse button is held
            paint_type = cells[timestep % 2][col][row];
          }
        }
      }
      else if (event.type == sf::Event::KeyPressed)
      {
        switch (event.key.code)
        {
        case sf::Keyboard::C:
        case sf::Keyboard::R:
          // Reset the field on the press of the "C" or "R" key
          clear();
          break;
        case sf::Keyboard::Q:
          // Exit on the press of the "q" key
          return teardown();
        case sf::Keyboard::Right:
          // Step once when the right arrow key is pressed
          step();
          break;
        }
      }
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
    {
      // Continuously simulate when the spacebar is held down
      step();
    }

    if (sf::Mouse::isButtonPressed(sf::Mouse::Right))
    {
      sf::Vector2i MousePos = sf::Mouse::getPosition(window);

      // Extract the coordinates of the event
      int x = MousePos.x;
      int y = MousePos.y;

      // If the event is valid,
      if (within_bounds(x, y))
      {
        // Compute the coordinates of the cell which was clicked
        int col = floor(x / (float)width * cols);
        int row = floor(y / (float)height * rows);

        // Set the state of the selected cell
        cells[timestep % 2][col][row] = paint_type;
      }
    }
  }
  return 0;
}
