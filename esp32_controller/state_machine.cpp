#include "state_machine.h"
#include "comms.h"
#include "outputs.h"
#include "led.h"
#include "button.h"

static const unsigned long COMMAND_TIMEOUT_MS = 200;

static SystemState state = SystemState::NoWifi;
static bool armed = false;
static unsigned long lastCommandMs = 0;
static Command lastCommand;
static bool failsafeApplied = false;

static const char* stateName(SystemState s) {
  switch (s) {
    case SystemState::NoWifi:        return "NoWifi";
    case SystemState::WifiConnected: return "WifiConnected";
    case SystemState::Receiving:     return "Receiving";
  }
  return "?";
}

static LedPattern ledForState(SystemState s, bool isArmed) {
  switch (s) {
    case SystemState::NoWifi:        return LedPattern::SlowBlink;
    case SystemState::WifiConnected: return LedPattern::FastBlink;
    case SystemState::Receiving:     return isArmed ? LedPattern::SolidOn : LedPattern::RapidBlink;
  }
  return LedPattern::Off;
}

void stateMachineBegin() {
  state = SystemState::NoWifi;
  armed = false;
  lastCommandMs = 0;
  failsafeApplied = false;
}

void stateMachineUpdate() {
  if (buttonWasPressed()) {
    armed = !armed;
    Serial.println(armed ? "Armed." : "Disarmed.");
  }

  Command cmd;
  bool newCommand = commsTakeCommand(cmd);
  if (newCommand) {
    lastCommandMs = millis();
    lastCommand = cmd;
  }

  bool wifi     = commsWifiConnected();
  bool messages = lastCommandMs > 0 && (millis() - lastCommandMs) < COMMAND_TIMEOUT_MS;

  SystemState next;
  if      (!wifi)     next = SystemState::NoWifi;
  else if (!messages) next = SystemState::WifiConnected;
  else                next = SystemState::Receiving;

  if (next != state) {
    Serial.print("State: ");
    Serial.println(stateName(next));
    state = next;
  }

  if (state == SystemState::Receiving && armed) {
    if (newCommand) {
      outputsApply(lastCommand);
    }
    failsafeApplied = false;
  } else if (!failsafeApplied) {
    outputsFailsafe();
    failsafeApplied = true;
  }

  ledSet(ledForState(state, armed));
  commsSendStatus(stateName(state), armed, lastCommand.seq);
}

SystemState stateMachineGetState() { return state; }
bool stateMachineIsArmed()         { return armed; }
