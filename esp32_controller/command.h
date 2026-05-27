#ifndef COMMAND_H
#define COMMAND_H

#include <Arduino.h>

enum class GyroMode { Soft, Hard };

struct Command {
  float steering;  // -1.0 (left) to 1.0 (right)
  float throttle;  // -1.0 (brake/reverse) to 1.0 (forward)
  GyroMode gyroMode;
  float gyroGain;  // 0.0 to 1.0
  uint32_t seq;
};

#endif
