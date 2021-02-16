/*
* This is a personal academic project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com
*/
#include "rng.h"
#include <random>
#include <functional>

typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<float> Distribution;

auto uniform_generator = std::bind(Distribution(0.0f, 1.0f), Engine(1));

float rng(const float range_min, const float range_max) {
  return static_cast<float>( uniform_generator()) * (range_max - range_min) + range_min;
}