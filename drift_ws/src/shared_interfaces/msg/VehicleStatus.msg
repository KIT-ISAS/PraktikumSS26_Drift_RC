# Status reply from the vehicle ECU (ESP32).
# Published at ~10 Hz when the ECU is responding.

string state         # ECU state machine: "NoWifi" | "WifiConnected" | "Receiving"
bool armed           # True if the ECU is armed (outputs active)
uint32 last_seq      # Sequence number of the last accepted command
bool connected       # False if no status reply received within the timeout window;
                     # other fields are stale when connected is False.
