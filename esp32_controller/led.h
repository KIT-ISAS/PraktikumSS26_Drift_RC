#ifndef LED_H
#define LED_H

#include <Arduino.h>

enum class LedPattern {
  Off,
  SlowBlink,    // 1 Hz  — no WiFi
  DoublePulse,  //        — WiFi up, no commands
  RapidBlink,   // 8 Hz  — receiving, not armed
  SolidOn,      //        — receiving + armed
};

void ledBegin();
void ledSet(LedPattern pattern);
void ledUpdate();

#endif
