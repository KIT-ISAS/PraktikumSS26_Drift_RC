#ifndef STATE_MACHINE_H
#define STATE_MACHINE_H

#include <Arduino.h>

enum class SystemState {
  NoWifi,        // LED: slow blink   1 Hz
  WifiConnected, // LED: double pulse 1 Hz
  Receiving,     // LED: rapid blink (unarmed) or solid (armed)
};

void stateMachineBegin();
void stateMachineUpdate();
SystemState stateMachineGetState();
const char* stateMachineGetStateName();
bool stateMachineIsArmed();
uint32_t stateMachineGetLastSeq();

#endif
