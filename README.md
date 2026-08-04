

# Practical Course Machine Learning and Intelligent Systems Summer Semester 2026
In this university team project, our group was asked to teach an rc car to drift using rl, which is an interesting and fun challenge. We inherited a baseline setup from our previous group, with a messy hardware setup and custom software. We cleaned up the hardware on the drift chassis by introducing a single ESP32 microcontroller, a 3D printed mounting for the vr localization tracker, and a 3D printed bumper to protect the car from impacts. We wrote the software from scratch and set up a proper development environment with Docker, ROS2, Foxglove, a CI pipeline and a clean workflow to train and deploy vehicle models and rl agents. We trained a physics-informed model with real driving data and trained a sac rl agent on this vehicle model. While the agent performs flawlessly in the simulation environment, we could not completely close the sim-to-real gap. Missing wheel speed information made the vehicle model not accurate enough in high slip areas. Nevertheless, we set a solid foundation for the next group that will take this challenge.

Contributors: Julian Bihl, Louis Wagner, Wenyu Liu
E-Mail: ugggq@student.kit.edu, louis.wagner@student.kit.edu, uxmtk@student.kit.edu
Supervisor: Daniel Frisch


# Repository Structure

## Top level

| Path | Content |
|------|---------|
| `drift_ws/` | The colcon workspace — all ROS 2 packages (`src/`), top-level launch files (`launch/`) and training runs (`runs/`) |
| `esp32_controller/` | Arduino firmware for the car's control ECU (ESP32): UDP/JSON command input, state machine, arming, PWM output to servo/ESC. [README.md](esp32_controller/README.md) is the authoritative spec |
| `steamvr_bridge/` | Host-side script (runs *outside* the container) that reads tracker poses from SteamVR via OpenVR and streams them over UDP to `steamvr_pkg` |
| `documentation/` | Setup and tooling guides (`Docker.md`, `Ros.md`, `Python.md`, `Ruff.md`, `Foxglove.md`) plus the final report and presentations |
| `.devcontainer/` | Docker dev container definition, compose files (incl. GPU variant), and post-start scripts |
| `.github/workflows/` | CI pipeline (`ros2_ci.yml`): build, `colcon test`, Ruff check/format |
| `foxglove/` | Foxglove Studio layout/config for visualizing the running system |
| `drift_ws/runs/` | Training run directories (checkpoints, configs, metrics), git-ignored: `agent/` for RL policies (plus `agent/analysis/` summaries), `vehicle_model/` for dynamics models |
| `pyproject.toml` | Ruff config (incl. the auto-generated first-party package list) |

## ROS 2 packages (`drift_ws/src/`)

| Package | Build type | Role |
|---------|-----------|------|
| `shared_interfaces` | ament_cmake | Custom messages/services: `VehicleActionRequest`, `VehicleStatus`, `Track`, `TrackProgress`, `ConnectionRequest` |
| `shared_pkg` | ament_python | Cross-package constants: the `Topics` registry and `frames` (TF frame names) |
| `joystick_pkg` | ament_python | Reads the controller via the ROS `joy` node and maps inputs to `VehicleActionRequest` |
| `vehicle_communication_pkg` | ament_python | Bridges ROS ↔ car over a pluggable `CommunicationInterface` (UDP/JSON to the ESP32); republishes `VehicleStatus` |
| `libsurvive_pkg` | ament_cmake (C++) | Tracking source A: wraps the `libsurvive_ros2` driver, publishes tracker and lighthouse poses in its own world frame |
| `steamvr_pkg` | ament_python | Tracking source B: receives poses from the host-side `steamvr_bridge` over UDP and publishes the same TF contract plus a pseudo-IMU |
| `car_localization_pkg` | ament_python | Source-agnostic localization: anchors the world frame against the lighthouse constellation and derives the canonical `/car/pose`, `/car/odom`, `/car/imu` at the CoG in ISO 8855 |
| `track_pkg` | ament_python | ROS wrapper around the Bézier track: publishes the latched centerline, lap/progress tracking, and Foxglove markers |
| `simulation_pkg` | ament_python | Runs a `drift_core` vehicle model as a drop-in replacement for the real car — same `/car/pose`, `/car/odom`, TF surface |
| `agent_pkg` | ament_python | Control sources beyond the joystick: the trained RL agent node, an MPC node, and data-collection / demo-recording nodes |
| `drift_core` | plain pip package | ROS-free core library (see below) — imported by `track_pkg`, `simulation_pkg` and `agent_pkg`, and usable standalone on a GPU box |

## `drift_core` modules

| Module | Content |
|--------|---------|
| `models/` | Vehicle dynamics models: `kinematic`, `dynamic`, `arcade`, `neural`, `hybrid`, plus ESC model, integrators and a config-driven `loader` |
| `environments/` | Gymnasium drift environment, observation builder, reward functions and env configs |
| `agents/` | RL policy plus the `train_agent` / `finetune_agent` / `evaluate_agent` / `analyze_runs` entry points |
| `training/` | Vehicle-model training: trainer, config, run-directory handling |
| `data/` | Rosbag reader, dataset assembly, preprocessing, driving-regime labelling and sanity checks |
| `evaluation/` | Metrics, plots and MCAP writing for evaluating models and agents |
| `track/` | `BezierTrack` geometry and the `TrackProjector` (arc-length, lateral error, lap counting) |
| `cars/`, `tracks/` | YAML parameter sets: the drift RC car, and the figure-8 track |
