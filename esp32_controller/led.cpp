#include "led.h"

// GPIO 2 is the built-in LED on most ESP32 DevKit boards.
static const int LED_PIN = 2;

static LedPattern currentPattern = LedPattern::Off;
static unsigned long patternStartMs = 0;

void ledBegin() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
}

void ledSet(LedPattern pattern) {
  if (pattern == currentPattern) return;
  currentPattern = pattern;
  patternStartMs = millis();
}

void ledUpdate() {
  unsigned long phase = millis() - patternStartMs;
  bool on = false;

  switch (currentPattern) {
    case LedPattern::Off:
      break;
    case LedPattern::SlowBlink:
      on = (phase % 1000) < 500;
      break;
    case LedPattern::DoublePulse:
      // 100ms on, 100ms off, 100ms on, 700ms off (1 s total)
      { unsigned long p = phase % 1000;
        on = p < 100 || (p >= 200 && p < 300); }
      break;
    case LedPattern::RapidBlink:
      on = (phase % 125) < 62;
      break;
    case LedPattern::SolidOn:
      on = true;
      break;
  }

  digitalWrite(LED_PIN, on ? HIGH : LOW);
}
