#include "deck.hpp"

#include <iostream>

int main() {
  Deck deck;
  
  std::cout << deck << '\n';

  deck.n_shuffle_riffle(3);

  std::cout << deck << '\n';
}
