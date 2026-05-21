#ifndef STATE_MACHINE_H
#define STATE_MACHINE_H

#include <Arduino.h>

enum class SystemState {
  NoWifi,        // LED: slow blink  1 Hz
  WifiConnected, // LED: fast blink  4 Hz
  PeerConnected, // LED: double-pulse
  Receiving,     // LED: rapid blink (unarmed) or solid (armed)
};

void stateMachineBegin();
void stateMachineUpdate();
SystemState stateMachineGetState();
bool stateMachineIsArmed();

#endif
