#include "button.h"

// GPIO 0 is the BOOT button on most ESP32 DevKit boards (active low).
static const int BUTTON_PIN    = 0;
static const unsigned long DEBOUNCE_MS = 50;

static bool stableState  = HIGH;
static bool lastReading  = HIGH;
static unsigned long lastChangeMs = 0;
static bool pressEvent = false;

void buttonBegin() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
}

void buttonUpdate() {
  pressEvent = false;
  bool reading = digitalRead(BUTTON_PIN);
  unsigned long now = millis();

  if (reading != lastReading) {
    lastChangeMs = now;
    lastReading = reading;
  }

  if ((now - lastChangeMs) >= DEBOUNCE_MS && reading != stableState) {
    stableState = reading;
    if (stableState == LOW) pressEvent = true;  // active-low: LOW = pressed
  }
}

bool buttonWasPressed() {
  return pressEvent;
}
