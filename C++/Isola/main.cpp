/*
 * Benjamin Chislett's Starter Code
 * Isola Gameplay Simulator
 *
 * See the original game here:
 * https://www.lexaloffle.com/bbs/?pid=41361
 *
 * Open License, do anything you want with this code
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <tuple>
#include <random>

// Setting up the board structure
constexpr int board_size = 5;

struct BoardState {
  int pos[2][2];
  bool holes[board_size][board_size];
  int turn_count;
};

// Random Number Generator
static std::random_device rd;
static std::mt19937 gen(rd());

// Logger for std::cout, set to QUIET to skip the board printing step
// Useful when running many games automatically
constexpr bool QUIET = false;

struct Log {
  bool ignore_quiet;

  Log(bool always = false) : ignore_quiet(always) {}

  template<typename T>
  Log &operator<<(const T &v) {
    if (!QUIET || ignore_quiet) {
      std::cout << v;
    }
    return *this;
  }
};

// Big ugly board printer
void print_board(const BoardState &state) {
  Log() << '\n';

  for (int i = 0; i < 2 * board_size + 4; i++) {
    if (i >= 4 && (i - 4) % 2 == 0 && (i - 4) / 2 <= board_size)
      Log() << (i - 4) / 2 + 1;
    else Log() << ' ';
  }
  Log() << '\n';
    
  for (int i = 0; i < 3; i++) Log() << ' ';
  for (int i = 3; i < 2 * board_size + 4; i++) Log() << '-';
  Log() << '\n';

  for (int row = 0; row < board_size; row++) {
    Log() << row+1 << "  " << "|";
    for (int col = 0; col < board_size; col++) {
      char pos = ' ';
      if (state.holes[row][col]) pos = 'X';
      else if (row == state.pos[0][0] && col == state.pos[0][1]) pos = 'A';
      else if (row == state.pos[1][0] && col == state.pos[1][1]) pos = 'B';

      Log() << pos;
      Log() << "|";
    }
    Log() << '\n';

    for (int i = 0; i < 3; i++) Log() << ' ';
    for (int i = 3; i < 2 * board_size + 4; i++) Log() << '-';
    Log() << '\n';
  }
}

// Move logistics

bool check_move(const BoardState &state, int player, int row, int col) {
  return (row >= 0 && row < board_size)
      && (col >= 0 && col < board_size)
      && (std::abs(state.pos[player][0] - row) <= 1)
      && (std::abs(state.pos[player][1] - col) <= 1)
      && (!(state.pos[player][0] == row && state.pos[player][1] == col))
      && (!(state.pos[1-player][0] == row && state.pos[1-player][1] == col))
      && (!(state.holes[row][col]));
}

bool check_bomb(const BoardState &state, int row, int col) {
  return (row >= 0 && row < board_size)
      && (col >= 0 && col < board_size)
      && (!(state.pos[0][0] == row && state.pos[0][1] == col))
      && (!(state.pos[1][0] == row && state.pos[1][1] == col))
      && (!(state.holes[row][col]));
}

BoardState bomb(const BoardState &state, int row, int col) {
  BoardState newstate = state;
  newstate.holes[row][col] = true;
  return newstate;
}

BoardState move(const BoardState &state, int player, int row, int col) {
  BoardState newstate = state;
  newstate.pos[player][0] = row;
  newstate.pos[player][1] = col;

  if (state.turn_count == 0) {
    newstate = bomb(newstate, state.pos[player][0], state.pos[player][1]);
  }

  return newstate;
}

std::vector<std::pair<int, int> > get_moves(const BoardState &state, int player) {
  std::vector<std::pair<int, int> > legal_moves;

  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      if (check_move(state, player, state.pos[player][0] + dy, state.pos[player][1] + dx))
        legal_moves.emplace_back(state.pos[player][0] + dy, state.pos[player][1] + dx);
    }
  }

  return legal_moves;
}

int count_moves(const BoardState &state, int player) {
  return get_moves(state, player).size();
}

std::vector<std::pair<int, int> > get_bombs(const BoardState &state) {
  std::vector<std::pair<int, int> > legal_bombs;

  for (int i = 0; i < board_size; i++) {
    for (int j = 0; j < board_size; j++) {
      if (check_bomb(state, i, j))
        legal_bombs.emplace_back(i, j);
    }
  }

  return legal_bombs;
}

// Approximate evaluation heuristic for a rudimentary AI
float heuristic(const BoardState &state, int player) {
  return count_moves(state, player) - count_moves(state, 1 - player);
}

// AI's simple logic, missing a search tree!
std::pair<int, int> ai_make_move(const BoardState &state, int player) {
  auto legal_moves = get_moves(state, player);

  float minval = 999;
  std::pair<int, int> best_move;
  for (auto m : legal_moves) {
    float curr = heuristic(move(state, player, m.first, m.second), 1 - player);
    if (curr < minval) {
      best_move = m;
      minval = curr;
    }
  }

  return best_move;
}

std::pair<int, int> ai_make_bomb(const BoardState &state, int player) {
  auto legal_bombs = get_moves(state, player);

  auto distr = std::uniform_int_distribution<>(0, legal_bombs.size() - 1);
  return legal_bombs[distr(gen)];
}

int main() {
  bool AI_controls[2] = {false, true};

  BoardState state = {0};
  state.pos[0][0] = 0;
  state.pos[0][1] = board_size / 2;
  state.pos[1][0] = board_size - 1;
  state.pos[1][1] = board_size / 2;

  print_board(state);
  
  int player = 0;
  int row, col;
  while (1) {
    if (count_moves(state, player) == 0) {
      Log(true) << "Game over! Player " << ((1 - player) == 0 ? 'A' : 'B') << " wins!\n";
      return 0;
    }

    if (!AI_controls[player]) {
      std::cin >> row >> col;
      row--; col--;
    } else {
      std::tie(row, col) = ai_make_move(state, player);
    }

    if (!check_move(state, player, row, col)) {
      Log(true) << "\nIllegal move! Try again.\n";
      continue;
    }

    state = move(state, player, row, col);

    print_board(state);

    do {
      if (!AI_controls[player]) {
        std::cin >> row >> col;
        row--; col--;
      } else {
        std::tie(row, col) = ai_make_bomb(state, player);
      }

      if (!check_bomb(state, row, col)) {
        Log(true) << "\nIllegal bomb position. Try again.\n";
      }
    } while (!check_bomb(state, row, col));

    state = bomb(state, row, col);

    print_board(state);

    player = 1 - player;

    if (player == 0)
      state.turn_count++;
  }
}

