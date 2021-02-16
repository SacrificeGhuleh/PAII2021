/*
* This is a personal academic project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com
*/
#include "timer.h"

Timer::Timer() : beg_(std::chrono::high_resolution_clock::now()) {}

void Timer::reset() { beg_ = std::chrono::high_resolution_clock::now(); }

double Timer::elapsed() const {
  return std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>
      (std::chrono::high_resolution_clock::now() - beg_).count();
}
