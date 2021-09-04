#pragma once

#include "card.hpp"

#include <random>
#include <vector>
#include <iostream>

struct Deck {
  std::vector<Card> cards;
  std::vector<Card> discard;

  Deck() {
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 13; j++) {
        cards.push_back(std::make_pair(i, j));
      }
    }
  }

  void shuffle_riffle() {
    std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution dist;
    auto random = [&](){ return dist(rng); };

    int length = cards.size();
    int approx_middle = (length / 2) + (random() * 2 - 1) * (length * 0.05);

    std::vector<Card> new_cards;
    int bottom = 0;
    int top = 0;

    while (bottom + top < length) {
      bool which = random();
      if (which && (length - top - 1) >= approx_middle) {
        new_cards.push_back(cards[approx_middle + top++]);
      }

      if (!which && bottom < approx_middle) {
        new_cards.push_back(cards[bottom++]);
      }
    }

    cards = new_cards;
  }

  void n_shuffle_riffle(int n) {
    for (int i = 0; i < n; i++) shuffle_riffle();
  }

  friend std::ostream& operator<<(std::ostream& os, const Deck& deck);
};

std::ostream& operator<<(std::ostream& os, const Deck& deck) {
  os << "Deck: ";
  for (auto card : deck.cards) os << card << ", ";

  os << "\nDiscards: ";
  for (auto card : deck.discard) os << card << ", ";

  return os;
}
