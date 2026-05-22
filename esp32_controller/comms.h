#ifndef COMMS_H
#define COMMS_H

#include <Arduino.h>
#include "command.h"

void commsBegin();
void commsUpdate();
bool commsWifiConnected();
bool commsTakeCommand(Command& out);
void commsSendStatus(const char* state, bool armed, uint32_t lastSeq);

#endif
