# Predecessor models (WS 25/26)

These models are from the Winter term 25/26 predecessor project
(`PraktikumWS25_Autonomous_Drift_RC/reward_environment/model/`), adapted to our
current code base so they plug into the `VehicleModel` interface like any other
model (`build_model("old_standard")` / `build_model("old_slip_angle")`).

| registry name | class | original file |
|---|---|---|
| `old_standard` | `OldStandardVehicleModel` | `standard_vehicle.py` (`Vehicle`) |
| `old_slip_angle` | `OldSlipAngleVehicleModel` | `slip_angle_vehicle.py` (`SlipAngleVehicle`) |

## What was adapted

The dynamics equations are ported 1:1; the surrounding plumbing changed:

* **Interface.** The originals were mutable pygame `Controllable` objects
  (`update(dt)` + setters). The ports are `VehicleModel` dataclasses with
  `step(state, action, dt)`; pose/velocity go through `VehicleState` (body-frame
  `vx`/`vy`, ISO 8855). The slip-angle model's drift state machine (slip,
  drifting/spin-out/reverse flags) stays inside the instance and carries across
  steps, so one instance must be stepped sequentially.
* **Units.** The originals worked in pixels (55 px car, 800 px/s top speed).
  All defaults are converted to SI using the scale implied by the real car's
  0.256 m wheelbase (slip-angle model: 1 px ≈ 6.21 mm; standard: 1 px ≈ 5.17 mm,
  their wheelbase fractions differed). Original values are noted next to each
  field.
* **Timestep.** The predecessor game ran at a fixed 8 ms tick and several
  updates were per-frame (no `dt`): the drift-mode slip-angle nudges and the
  whole spin-out routine. Those are converted to per-second rates (divided by
  0.008 s), so at `dt = 0.008` the ports match the originals step for step.
* Steering/geometry conventions already matched (positive steering = positive
  yaw = left in our y-up world frame).

## Bugs found in the originals (fixed in the port)

* `standard_vehicle.py`: leftover debug `print(abs(self.speed))` every step —
  removed.
* `standard_vehicle.py`: the forward vector's renormalization was computed
  (`length = math.hypot(...)`) but never applied, so the heading vector's
  magnitude could drift and scale the speed. Moot here: the port stores the
  heading as an angle.
* `standard_vehicle.py`: when braking out of forward motion, rolling friction
  was subtracted twice (lines 50–51). The port applies it once.
* `standard_vehicle.py`: the comment claimed `max_steering_angle` is in degrees,
  but the math (`tan(steering_angle)`) requires radians (and the world set it
  with `math.radians(30)`). The port documents radians.
* `slip_angle_vehicle.py`: frame-rate dependence described above (per-frame
  updates without `dt`) — converted to rates.
* `slip_angle_vehicle.py`: `previous_throttle_input` was stored but never read;
  `stop_car_forward_motion` read `self.forward` before it necessarily existed.
  Both dropped (not part of the `VehicleModel` interface).

## Known quirks kept for faithfulness

* The slip-angle model never enforces its `max_speed`; top speed emerges from
  the engine/drag balance (≈ 8 m/s at full throttle with full slip).
* Full lock + full throttle drifts very aggressively and spin-outs are drastic
  (up to ~9 rad/s drift yaw; the spin-out yaw rate scales with speed) — that is
  how the original behaved at its 8 ms tick.
* Near standstill in drift mode the velocity "direction" degenerates to the raw
  velocity vector, effectively stopping the car (original behavior).
