# Testing without the car

Most ROS-side changes can be verified end to end without the RC car, the ESP32 or the
Vive setup. The ESP32 speaks plain JSON over UDP, so a small script can stand in for it
on loopback and the real nodes run unmodified against it.

This complements the two things that already exist:

- `colcon test` — unit tests, including `vehicle_communication_pkg`'s UDP tests.
- `esp32_controller/test_esp32.py` — drives a **real** ESP32 from the host. This
  document is the mirror image: a fake ESP32 driven by the **real ROS nodes**.

### Build and source

From `drift_ws/`:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select shared_interfaces shared_pkg vehicle_communication_pkg
source install/setup.bash
```

### A fake ESP32 on loopback

The wire format is JSON datagrams on port 12345; `esp32_controller/README.md` is the
authoritative spec. A stand-in needs to do only three things:

1. Bind `127.0.0.1:12345`.
2. Remember the address the last command arrived from — the node reveals its reply
   address only by sending, so there is nothing to reply to before that.
3. Reply with a status datagram at 10 Hz:

```json
{"state": "Receiving", "armed": false, "last_seq": 42}
```

Roughly thirty lines of Python with `socket` covers it.

### Driving it

```bash
ros2 run vehicle_communication_pkg vehicle_communication_node --ros-args -p ip_address:=127.0.0.1
```

The node only transmits once commanded, so publish something for it to forward:

```bash
ros2 topic pub -r 20 /car/vehicle_action_request shared_interfaces/msg/VehicleActionRequest \
  "{steering: 0.0, throttle: 0.0, gyro_mode: soft, gyro_gain: 0.0}"
```

### What to check

```bash
ros2 topic echo /car/vehicle_status --once
ros2 topic hz /car/vehicle_status
```

`state`, `armed` and `last_seq` should mirror what the fake is sending, and `last_seq`
should track the commands being published.

The useful failure-mode probe is to kill the fake: `connected` must drop to `false`,
and `state`, `armed` and `last_seq` are reset to `""`, `false` and `0` rather than left
stale.

### Checking firmware without flashing

`arduino-cli` is not installed in the devcontainer — firmware compiles happen on the
host. To catch C++ syntax errors in a new firmware module from inside the container, a
`g++ -fsyntax-only` pass against stub `Arduino.h` / `Wire.h` headers is usually enough.
