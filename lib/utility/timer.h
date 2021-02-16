/*
* This is a personal academic project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com
*/

#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
  Timer();
  
  void reset();
  
  double elapsed() const;

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> beg_;
};


#endif //TIMER_H
