# Vehicle models

Pluggable vehicle dynamics models for the simulation + RL pipeline. Every model
implements one interface and is created by name through a factory, so the
simulator, training, and evaluation code stays model-agnostic — swapping and
comparing models is the core experiment loop.

## The interface

- **`VehicleState`** — `x, y, heading` (world pose) + `vx, vy, yaw_rate` (body-frame
  velocities), plus derived accessors `speed` (V) and `sideslip` (β) for consumers
  like the RL observation. **`VehicleAction`** — `steering, throttle`, both in `[-1, 1]`
  (normalized; the real car exposes no direct steering angle or wheel torque, so
  models learn/approximate those mappings internally).
- **`VehicleModel`** (ABC): `step(state, action, dt) -> state` is the one required
  method (deterministic); `step_stochastic(...)` defaults to it. Basic hand-written
  models (`kinematic`, `arcade`) subclass this directly.
- **`TrainableModel(VehicleModel)`**: torch models implement
  `derivative(dynamic, action) -> Prediction` — the **timestep-free** per-second
  derivative of `[vx, vy, yaw_rate]` (the differentiable source of truth used by
  training). The framework integrates it: `step` and the training rollout both do
  `next = state + derivative·dt` (Euler / Euler–Maruyama), so models never handle
  `dt` or pose themselves. They also provide `trainable_parameters()`,
  `to_checkpoint()` / `from_checkpoint()`.
- **`Prediction(mean, cov)`** unifies deterministic (`cov=None`) and stochastic
  (per-second covariance) outputs.
- **`build_model(name, ...)`** resolves a registry name, a run directory, or a
  `.pt` checkpoint:
  ```python
  build_model("arcade")                  # basic, hand-tuned
  build_model("kinematic")               # no-slip baseline
  # neural / dynamic / hybrid and run-dir loading ("runs/<id>/") are added later
  ```

## Models

| name | kind | torch | drifts? | use for |
|------|------|-------|---------|---------|
| `kinematic` | kinematic bicycle (no slip) | no | no (`vy ≡ 0`) | fast baseline / RL bootstrap / sanity |
| `arcade` | game-like, velocity-space | no | yes (tunable `grip`) | demos / feel-test in sim |

### `kinematic` — `KinematicBicycleModel`
No-slip rear-wheel bicycle. Speed from `throttle·max_accel − drag·speed` (clamped to
`[0, max_speed]`); `yaw_rate = speed / wheelbase · tan(steering·max_steer_angle)`;
midpoint heading integration. `vy` is always `0` — the car only moves along its
heading, so it **cannot drift**. Deterministic, with optional Gaussian noise
(`noise_vx`, `noise_yaw_rate`) via `step_stochastic`.
Params: `max_accel, max_speed, drag, wheelbase, max_steer_angle`.

### `arcade` — `ArcadeDriftModel`
Game-like drift model where body-frame `(vx, vy)` is the state. Throttle drives
`vx` (with drag); steering yaws the car (scaled by speed, so it can't turn while
parked); the velocity vector is then rotated into the freshly-yawed body frame —
turning faster than the velocity follows is what creates the sideways slide — and
`grip` bleeds that lateral velocity away. Low `grip` → longer, looser slides; high
`grip` ≈ kinematic. Deterministic. Params: `max_speed, max_reverse_speed,
max_accel, drag, max_yaw_rate, steer_speed_full, grip, lateral_drag`.

> Learned / physical models (`neural`, `dynamic`, `hybrid`) are documented in the
> sections added with their respective PRs.

## Adding a new model

1. Hand-written model: subclass `VehicleModel` and implement `step(...)` (numpy).
   Learned / physics model: subclass `TrainableModel` and implement
   `derivative(dynamic, action) -> Prediction` plus `trainable_parameters()` /
   `to_checkpoint()` / `from_checkpoint()` — `step` is inherited.
2. Register it in `models/registry.py` (`MODEL_REGISTRY`, lazy getter) and, if it
   needs special construction, add a branch in `models/__init__.build_model`.
3. Add unit tests and a row to the table above.
