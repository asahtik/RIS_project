#pragma once

#include <sstream>

namespace colorFromHSV {
  enum {RED = 1, YELLOW, GREEN, CYAN, BLUE, MAGENTA} colors;

  /**
   * @brief Gets most likely colour from histogram
   * 
   * @param hist histogram of colours
   * @param num_clrs size of histogram
   * @return int enum
   */
  int get_from_hist(int* hist, int num_clrs) {
    // TODO
    return 0;
  }

  /**
   * @brief Get colour from avg hue
   * 
   * @param hue avg hue
   * @return int enum
   */
  int get_from_hue(float hue) {
    if(hue <= 60.0) return RED;
    else if(hue <= 120.0) return YELLOW;
    else if(hue <= 180.0) return GREEN;
    else if(hue <= 240.0) return CYAN;
    else if(hue <= 300.0) return BLUE;
    else return MAGENTA;
  }

  std::string enumToString(int e) {
    switch(e) {
    case RED:
      return "red";
      break;
    case YELLOW:
      return "yellow";
      break;
    case GREEN:
      return "green";
      break;
    case CYAN:
      return "cyan";
      break;
    case BLUE:
      return "blue";
      break;
    case MAGENTA:
      return "magenta";
      break;
    default:
      return "unknown";
      break;
    }
  }
}