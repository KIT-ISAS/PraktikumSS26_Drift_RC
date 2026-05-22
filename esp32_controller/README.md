# ESP32 Driftcar Controller

Firmware for the ESP32 control ECU.
Receives steering and throttle commands over WiFi/TCP and drives the car's servos and ESC via PWM.

## Hardware wiring

| Signal   | GPIO | Connected to         | PWM range        |
|----------|------|----------------------|------------------|
| Steering | 25   | Steering servo       | 1100 – 1900 µs   |
| Throttle | 26   | ESC                  | 1200 – 1800 µs   |
| Gyro     | 27   | Gyro sensitivity pin | 1100 – 1950 µs   |
| LED      | 2    | Built-in LED (PCB)   | —                |
| Button   | 0    | BOOT button (PCB)    | —                |

GPIO 25, 26, 27 are on the VIN side of the 38-pin DOIT DevKit. GPIO 2 and 0 are PCB-mounted and require no side pin.

Neutral (failsafe) for all outputs is 1500 µs. All PWM signals run at 50 Hz.

## State machine

The firmware runs a four-level state machine. Outputs are only active when all
levels are satisfied **and** the car is armed.

```
NoWifi ──wifi connects──▶ WifiConnected ──peer connects──▶ PeerConnected ──messages OK──▶ Receiving
   ◀──wifi drops──────────────────────────────────────────────────────────────────────────────┘
                    ◀──peer disconnects──────────────────────────────────────┘
                                                         ◀──message timeout──┘
```

| State          | Outputs      | LED                        |
|----------------|--------------|----------------------------|
| NoWifi         | Failsafe     | Slow blink (1 Hz)          |
| WifiConnected  | Failsafe     | Fast blink (4 Hz)          |
| PeerConnected  | Failsafe     | Double-pulse (2× per sec)  |
| Receiving      | Failsafe     | Rapid blink (8 Hz)         |
| Receiving + **armed** | Active | **Solid on**          |

Message timeout is 150 ms — if no valid command arrives within that window the
state drops back to `PeerConnected` and outputs go to failsafe.

## Arming

Press the **BOOT button** (GPIO 0) to toggle arm/disarm. The car will not move
until armed, even if all connectivity conditions are met. Arm state persists
across transient connection drops.

## Network / TCP protocol

- **Static IP:** `10.42.0.99` (subnet `10.42.0.0/24`, gateway `10.42.0.1`)
- **Port:** `12345` (TCP)
- **Protocol:** newline-delimited JSON, one command per line

### Command format

```json
{"steering": 0.0, "throttle": 0.0, "gyro_mode": "soft", "gyro_gain": 0.0, "seq": 1}
```

| Field       | Type   | Range        | Description                                      |
|-------------|--------|--------------|--------------------------------------------------|
| `steering`  | float  | −1.0 … 1.0   | Left (−1) to right (+1)                          |
| `throttle`  | float  | −1.0 … 1.0   | Reverse/brake (−1) to forward (+1)               |
| `gyro_mode` | string | `soft`/`hard`| Gyro correction direction                         |
| `gyro_gain` | float  | 0.0 … 1.0    | Gyro correction strength                         |
| `seq`       | uint32 | —            | Sequence number (for debugging, not validated)   |

Values outside the stated ranges are clamped silently. Only one TCP client is
served at a time. Send commands at ≥ 10 Hz to stay in `Receiving` state (the
timeout is 150 ms).

## WiFi networks (priority order)

Edit the arrays in `comms.cpp` to change credentials or add networks:

```cpp
static const char* WIFI_SSIDS[]     = {"JulianCar", "wifi_mid", "wifi_low"};
static const char* WIFI_PASSWORDS[] = {"Hund1234",  "pass_mid", "pass_low"};
```

The firmware tries networks in order, cycling through them every 8 s until one
connects. Reconnection is non-blocking — the LED and button keep working while
connecting.

## Dependencies (Arduino libraries)

- **ESP32Servo** — PWM output
- **ArduinoJson** — JSON parsing (bundled in `libraries/`)

Board: *ESP32 Dev Module* (or equivalent). Flash via Arduino IDE or `arduino-cli`.

## Test script

`test_esp32.py` tests all interaction modes and failure scenarios from any machine on the same network.

```bash
python test_esp32.py 10.42.0.99            # interactive menu
python test_esp32.py 10.42.0.99 all        # run every test in order
python test_esp32.py 10.42.0.99 timeout    # single test
```

| Test                  | What it provokes                              |
|-----------------------|-----------------------------------------------|
| `normal`              | Steady 20 Hz command stream                   |
| `timeout`             | Stop sending → failsafe after 500 ms          |
| `graceful_disconnect` | Clean TCP FIN                                 |
| `abrupt_disconnect`   | TCP RST (no FIN)                              |
| `reconnect`           | Disconnect then reconnect                     |
| `invalid_json`        | Malformed JSON line                           |
| `invalid_gyro_mode`   | Unknown `gyro_mode` string                    |
| `out_of_range`        | Values outside [−1, 1] (clamped, no error)    |
| `overflow`            | Line > 512 bytes without newline              |
| `no_newline`          | Valid JSON without trailing `\n`              |
| `rapid_cycles`        | 10 rapid connect/disconnect cycles            |
| `double_connect`      | Two clients simultaneously                    |

Each test prints the expected Serial output and LED behaviour before running.
