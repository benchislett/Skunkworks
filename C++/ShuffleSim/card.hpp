#pragma once

#include <iostream>
#include <utility>
#include <string>

const std::string SUITS[4] = {"\u2660", "\u2665", "\u2666", "\u2663"};
const std::string NUMBERS[13] = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"};

using Card = std::pair<int, int>;

std::ostream& operator<<(std::ostream& os, const Card& card) {
  os << NUMBERS[card.second] << SUITS[card.first];
  return os;
}
