#ifndef BUTTON_H
#define BUTTON_H

#include <Arduino.h>

void buttonBegin();
void buttonUpdate();
bool buttonWasPressed();  // true once per press event

#endif
