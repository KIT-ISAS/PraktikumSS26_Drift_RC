# RL Simulation Pipeline — Goals & Requirements

> Vision/requirements doc for the simulation + reinforcement-learning side of the project.
> Captures the intended workflow and design constraints so they don't have to be re-explained.
> The concrete implementation plan derived from this lives alongside it (see "Status" at the end).

## Goal

Train a reinforcement-learning (RL) agent **in simulation** that drives the RC car around a track
**while drifting**, then deploy that agent to the real car.

## System / hardware context

- **Car actuators:** a steering servo and an ESC for throttle **and** brake control.
- **Link:** both are commanded over **WiFi to an ESP32** on the car.
- **Compute:** the car's commands are calculated on an **external PC running ROS 2** (not on the car).
- **Manual control:** the car can be driven with an **Xbox controller**.
- **Localization:** the car carries an **HTC Vive tracker** (position, orientation, etc.).
- **Visualization:** **Foxglove** is used to visualize while driving.

## RL workflow (end to end)

1. **Record** one or more rosbags of the car being **driven and drifted manually**.
2. **Train a vehicle model** of the car from that recorded data **plus some measured properties**
   (physical parameters of the car).
3. **Train an RL agent in simulation** using that vehicle model — runs **in parallel**.
4. **Deploy** the trained agent to the real drift car.

Because it's uncertain whether this will work, the pipeline must support **experimentation**.

## Requirements / things we want to be able to do

- **Try different vehicle models and different training approaches** — easy to swap and compare.
- **Compare a vehicle model's predictions against the recorded driving** from the rosbags —
  ideally viewable **in Foxglove**.
- **Drive the car in simulation with the Xbox controller** using a vehicle model, to feel how it
  behaves.
- **See how an RL agent performs in simulation** via Foxglove, and **compare multiple agents
  side by side** in Foxglove.
- **Keep heavy ROS/Foxglove out of the training loop.** ROS + Foxglove are heavy; training
  (especially on a **GPU server**) should not run ROS the whole time. Saving **intermediate states**
  during model/agent training and inspecting them **afterwards** is acceptable / preferred.
- **No duplicated code between the ROS side and the training side.**

## Design constraints / decisions

- **Single ROS-free core library** is the source of truth for everything that must not depend on ROS
  (track geometry, vehicle models, the Gym environment, data pipeline, training). ROS packages are
  thin wrappers that import from it. This is what lets training run on a GPU server without ROS, and
  removes the duplication.
- **Track geometry (BezierTrack / TrackProjector) lives in the ROS-free core lib**; `track_pkg`'s ROS
  node imports it (decided — the geometry was previously duplicated and had already drifted).
- **Offline inspection = write MCAP files**, opened directly in Foxglove with **no ROS running**.
  Use the existing `rosbags` dependency to write poses / track / reward / agent-id during training
  and evaluation. (Decided over a live ROS replay node, for the GPU-server / no-ROS goal.)
- **Observation/action construction must be shared** between the training environment and the
  real-car deployment node, so the agent sees identical inputs in sim and reality (sim-to-real).

## Status

- Requirements captured here on 2026-06-15.
- Implementation approach: refactor the existing `vehicle_simulation` + `simulation_pkg` to the
  architecture above now, then build the comparison / multi-agent / deployment tooling in phases.
