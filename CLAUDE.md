# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

KIT research practicum "Autonomes Driften mit Reinforcement Learning" (PraktikumSS26_Drift_RC). The system tracks an RC car with a Vive lighthouse setup ("Holodeck"), drives it via an ESP32 control ECU, and (eventually) trains an RL policy to drift autonomously. See `Plan_01.md` for the project roadmap.

## Environment

Development happens inside the dev container (ROS 2 Jazzy base). All commands below assume you are in that container.

- ROS packages are installed globally; **pip packages live in the venv at `/opt/venv`** (Ubuntu forbids global pip installs). The venv is *not* activated — its site-packages is appended to `PYTHONPATH` (lower precedence than ROS) in the Dockerfile. This can cause conflicts for custom packages; see `documentation/Python.md`.
- Container lifecycle, joystick device passthrough, and VS Code remote setup: `documentation/Docker.md`.
- On container start, `.devcontainer/scripts/post_start.sh` runs `update_package_list.sh`, which **regenerates the first-party package list** in both the Ruff config (`pyproject.toml`) and the ament_flake8 config. Do not hand-maintain that list.

## Common commands

The colcon workspace root is `drift_ws/`. Run build/test commands from there.

```bash
# Source (already in container .bashrc)
source /opt/ros/jazzy/setup.bash
source /root/repo/drift_ws/install/setup.bash

# Build (from drift_ws/)
colcon build [--packages-select <pkg>]

# Run a node / launch
ros2 run <pkg> <executable>
ros2 launch drift_ws/launch/manual_driving.launch.py   # joystick + vehicle comms

# Tests (includes auto-generated linter tests)
colcon test --event-handlers console_direct+
colcon test --event-handlers console_direct+ --pytest-args '-m not linter'   # skip linters
colcon test --event-handlers console_direct+ --pytest-args '-m linter'       # linters only

# Lint / format (run from repo root)
ruff check .
ruff format src/        # CI enforces `ruff format --check src/`
```

To create a package, use `ros2 pkg create <name> --build-type ament_python` from `drift_ws/src`, then re-run `update_package_list.sh`.

## Linting: two tools must agree

Both **Ruff** (config in `pyproject.toml`) and **ament_flake8** (runs as part of `colcon test`) gate the code, and CI fails on either. The known conflict — Flake8 wants single quotes, Ruff uses double — is resolved by patching `ament_flake8.ini` to allow double quotes (`.devcontainer/scripts/init_flake8.sh`). When in doubt, match Ruff's output since format-on-save uses it. Details: `documentation/Ruff.md`, `documentation/Ros.md`.

## Architecture

ROS 2 packages live in `drift_ws/src/`:

| Package | Build type | Role |
|---------|-----------|------|
| `shared_interfaces` | ament_cmake | Custom msgs/srvs: `VehicleActionRequest`, `VehicleStatus`, `ConnectionRequest` |
| `shared_pkg` | ament_python | Cross-package constants — notably the `Topics` class (topic name registry) |
| `joystick_pkg` | ament_python | Reads controller via ROS `joy` node, maps inputs (`xbox_one_mapper`) to `VehicleActionRequest` |
| `vehicle_communication_pkg` | ament_python | Bridges ROS ↔ car: forwards `VehicleActionRequest` to the ESP32, republishes `VehicleStatus` |
| `holodeck_pkg` | ament_cmake (C++) | Thin wrapper around the `libsurvive_ros2` driver for Vive lighthouse tracking |
| `track_pkg` | ament_python | Publishes track boundary geometry |
| `example_pkg` | ament_python | Template/reference package |

**Control data flow:** `joystick_pkg` → `VehicleActionRequest` topic → `vehicle_communication_pkg` → UDP/JSON → ESP32 firmware → PWM (servo/ESC). Status flows back: ESP32 → UDP → `vehicle_communication_pkg` → `VehicleStatus` topic.

**Tracking data flow:** Vive lighthouses → `holodeck_pkg`/libsurvive → vehicle & lighthouse poses published in the **world frame** via TF.

### Conventions

- **Never hardcode topic names.** Import them from `shared_pkg`'s `Topics` class (e.g. `Topics.VEHICLE_ACTION_REQUEST`). Add new topics there.
- `vehicle_communication_pkg` talks to the car through a pluggable `CommunicationInterface` (`lib/`), with `UdpCommunicationClient` as the concrete implementation — keep transport details behind that interface.

## ESP32 firmware (`esp32_controller/`)

Arduino firmware for the car's control ECU, independent of the ROS build. It receives steering/throttle over UDP (JSON datagrams, port 12345, static IP `10.42.0.99`) and drives PWM outputs, gated by a three-level state machine plus an arm toggle. `esp32_controller/README.md` is the authoritative spec: wiring, UDP protocol, arming/reverse sequences, state machine, build/flash via `arduino-cli`, and the `test_esp32.py` exerciser. Any change to the UDP message shape must stay in sync with `vehicle_communication_pkg`.

## Visualization

Foxglove via the foxglove_bridge (`ws://localhost:8765`). See `documentation/Foxglove.md`.
