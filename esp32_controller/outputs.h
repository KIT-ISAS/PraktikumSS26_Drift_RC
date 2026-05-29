#ifndef OUTPUTS_H
#define OUTPUTS_H

#include <Arduino.h>
#include "command.h"

void outputsBegin();
void outputsApply(const Command& cmd);
void outputsFailsafe();
bool outputsReady();  // false during initial ESC arming window

#endif
