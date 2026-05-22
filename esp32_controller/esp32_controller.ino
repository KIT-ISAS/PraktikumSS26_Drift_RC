#include "comms.h"
#include "outputs.h"
#include "led.h"
#include "button.h"
#include "state_machine.h"


void setup() {
  Serial.begin(115200);
  delay(1000);
  ledBegin();
  buttonBegin();
  outputsBegin();
  commsBegin();
  stateMachineBegin();
}

void loop() {
  buttonUpdate();
  commsUpdate();
  stateMachineUpdate();
  ledUpdate();
}
