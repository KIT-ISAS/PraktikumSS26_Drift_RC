# Vehicle Communication Package

Handles communication between the ROS 2 workspace and the ESP32 vehicle controller.
Commands are sent as UDP datagrams; the ESP32 sends status replies back at 10 Hz.

## Network setup

The ESP32 uses a static IP (`10.42.0.99`, port `12345`). The PC and ESP32 must be on the same
WiFi network. Since the KIT network blocks these connections, use a mobile hotspot
(e.g. your laptop's hotspot or the USB stick on the holodeck PC). Configure the
SSID/password in `esp32_controller/comms.cpp`.

Set the ESP32's IP and port in the node parameters:

```yaml
# drift_ws/src/vehicle_communication_pkg/config/vehicle_communication_params.yaml
vehicle_communication_node:
  ros__parameters:
    ip_address: "10.42.0.99"
    port: 12345
```

## ROS interface

**Subscribed topic:** `/car/vehicle_action_request` (`shared_interfaces/VehicleActionRequest`)

```
float32 steering   # -1.0 (left) to 1.0 (right)
float32 throttle   # -1.0 (brake/reverse) to 1.0 (forward)
string  gyro_mode  # "soft" or "hard"
float32 gyro_gain  # 0.0 (off) to 1.0 (full)
```

**Services:**
- `connect` (`std_srvs/Trigger`) — open the UDP socket
- `disconnect` (`std_srvs/Trigger`) — close the UDP socket

The socket is opened automatically on startup. The node logs the ESP32's reported
state and arm status (received via status replies) whenever it changes.

## UDP protocol

Each command is a JSON datagram sent to the ESP32:

```json
{"session_id": 12345, "steering": 0.0, "throttle": 0.0, "gyro_mode": "soft", "gyro_gain": 0.0, "seq": 1}
```

`session_id` is a random 32-bit integer chosen once per process. When the ESP32 sees a new `session_id` it immediately resets its sequence window, so communication resumes with the very first packet after a node restart.

The ESP32 replies to the sender at 10 Hz:

```json
{"state": "Receiving", "armed": true, "last_seq": 42}
```

Status replies time out after 500 ms of silence — the node logs a warning when
this happens.

## Reconnection

The node uses a connectionless UDP socket, so no explicit reconnect is needed for normal WiFi drops — `sendto` simply resumes once the network is restored.

For persistent socket failures (e.g. `ENETDOWN`), `UdpCommunicationClient` tracks consecutive send errors and automatically recreates the socket after 10 consecutive failures (~0.5 s at 20 Hz send rate).

## Communication library

`vehicle_communication_pkg/lib/` contains:

| Module                    | Description                            |
|---------------------------|----------------------------------------|
| `communication_interface` | Abstract base class                    |
| `udp_communication`       | UDP client with background status receiver thread and automatic socket recovery |
