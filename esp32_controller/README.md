# ESP32 Driftcar Controller

Firmware for the ESP32 control ECU.
Receives steering and throttle commands over WiFi/UDP and drives the car's servos and ESC via PWM.

## Hardware wiring

| Signal   | GPIO | Connected to         | PWM range        |
|----------|------|----------------------|------------------|
| Steering | 25   | Steering servo       | 1100 – 1900 µs   |
| Throttle | 26   | ESC                  | 1200 – 1800 µs   |
| Gyro     | 27   | Gyro sensitivity pin | 1100 – 1900 µs   |
| LED      | 2    | Built-in LED (PCB)   | —                |
| Button   | 0    | BOOT button (PCB)    | —                |

GPIO 25, 26, 27 are on the VIN side of the 38-pin DOIT DevKit. GPIO 2 and 0 are PCB-mounted and require no wiring.

Neutral (failsafe) for all outputs is 1500 µs. All PWM signals run at 50 Hz.

## State machine

The firmware runs a three-level state machine. Outputs are only active when all levels are satisfied **and** the car is armed.

```
NoWifi ──wifi connects──▶ WifiConnected ──commands arrive──▶ Receiving
   ◀──wifi drops─────────────────────────────────────────────────┘
                          ◀──command timeout (200 ms)────────────┘
```

| State                  | Outputs  | LED                    |
|------------------------|----------|------------------------|
| NoWifi                 | Failsafe | Slow blink (1 Hz)      |
| WifiConnected          | Failsafe | Fast blink (4 Hz)      |
| Receiving              | Failsafe | Rapid blink (8 Hz)     |
| Receiving + **armed**  | Active   | **Solid on**           |

The command timeout is 200 ms — if no valid command arrives within that window the state drops back to `WifiConnected` and outputs go to failsafe.

## Arming

Press the **BOOT button** (GPIO 0) to toggle arm/disarm. The car will not move until armed, even if all connectivity conditions are met. Arm state persists across transient connection drops.

## Network / UDP protocol

- **Static IP:** `10.42.0.99` (subnet `10.42.0.0/24`, gateway `10.42.0.1`)
- **Port:** `12345` (UDP)
- **Protocol:** JSON datagrams, one command per datagram

### Command (PC → ESP32)

```json
{"steering": 0.0, "throttle": 0.0, "gyro_mode": "soft", "gyro_gain": 0.0, "seq": 1}
```

| Field       | Type   | Range         | Description                                    |
|-------------|--------|---------------|------------------------------------------------|
| `steering`  | float  | −1.0 … 1.0    | Left (−1) to right (+1)                        |
| `throttle`  | float  | −1.0 … 1.0    | Reverse/brake (−1) to forward (+1)             |
| `gyro_mode` | string | `soft`/`hard` | Gyro correction direction                      |
| `gyro_gain` | float  | 0.0 … 1.0     | Gyro correction strength                       |
| `seq`       | uint32 | —             | Sequence number (for debugging, not validated) |

Values outside the stated ranges are clamped silently.
Send commands at ≥ 5 Hz to stay in `Receiving` state (timeout is 200 ms).

### Status reply (ESP32 → PC)

The ESP32 sends a status datagram back to the most recent sender at 10 Hz:

```json
{"state": "Receiving", "armed": true, "last_seq": 42}
```

| Field      | Type   | Description                          |
|------------|--------|--------------------------------------|
| `state`    | string | Current state machine state          |
| `armed`    | bool   | Whether the car is armed             |
| `last_seq` | uint32 | Sequence number of last accepted command |

### Reverse / ESC arming

Most RC ESCs require a double-tap to engage reverse:
1. Throttle below neutral (braking)
2. Return to neutral
3. Throttle below neutral again → reverse engages

When driving with the joystick, this happens naturally. Automated scripts must perform this sequence explicitly (see `test_esp32.py`, Phase 3a–3c).

## WiFi networks (priority order)

Edit the arrays in `comms.cpp` to change credentials or add networks:

```cpp
static const char* WIFI_SSIDS[]     = {"JulianCar", "wifi_mid", "wifi_low"};
static const char* WIFI_PASSWORDS[] = {"...",        "...",      "..."};
```

The firmware tries networks in order, cycling every 8 s until one connects. Reconnection is non-blocking — the LED and button keep working while connecting.

## Dependencies (Arduino libraries)

Libraries are declared in `sketch.yaml` and must be installed before compiling.
They are not committed to the repository.

With `arduino-cli`:
```bash
arduino-cli lib install "ArduinoJson@7.4.3" "ESP32Servo@3.2.0"
```

Or install via **Arduino IDE → Tools → Manage Libraries**.

| Library     | Version | Purpose    |
|-------------|---------|------------|
| ArduinoJson | 7.4.3   | JSON parsing/serialization |
| ESP32Servo  | 3.2.0   | PWM output via LEDC hardware |

Board: *ESP32 Dev Module* (or equivalent). Flash via Arduino IDE or `arduino-cli`.

## Test script

`test_esp32.py` tests all interaction modes and failure scenarios from any machine on the same network.

```bash
python test_esp32.py 10.42.0.99            # interactive menu
python test_esp32.py 10.42.0.99 all        # run every test in order
python test_esp32.py 10.42.0.99 timeout    # single test
```

| Test               | What it provokes                                    |
|--------------------|-----------------------------------------------------|
| `normal`           | Full sweep: steering, throttle, reverse, combined   |
| `status`           | Print bidirectional status replies at 10 Hz         |
| `timeout`          | Stop sending → failsafe → resume                   |
| `reconnect`        | New sender socket after silence                     |
| `invalid_json`     | Malformed JSON datagram                             |
| `invalid_gyro_mode`| Unknown `gyro_mode` string                          |
| `out_of_range`     | Values outside [−1, 1] (clamped, no error)          |
| `overflow`         | Datagram > 511 bytes (discarded by firmware)        |
| `multiple_senders` | Two UDP sockets alternating                         |
| `wifi_loss`        | Manually disable/enable AP mid-stream               |

Each test prints the expected Serial output and LED behaviour before running.
