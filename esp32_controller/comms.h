#ifndef COMMS_H
#define COMMS_H

#include <Arduino.h>
#include "command.h"

void commsBegin();
void commsUpdate();
bool commsWifiConnected();
bool commsPeerConnected();
bool commsTakeCommand(Command& out);

#endif
