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
| `old_standard` | predecessor (WS 25/26) kinematic bicycle | no | no (`vy ≡ 0`) | comparison with predecessor sim |
| `old_slip_angle` | predecessor (WS 25/26) scripted drift model | no | yes (drift state machine) | comparison with predecessor sim |

### `kinematic` — `KinematicBicycleModel`
No-slip rear-wheel bicycle. Speed from `throttle·max_accel − drag·speed` (clamped to
`[0, max_speed]`); `yaw_rate = speed / wheelbase · tan(steering·max_steer_angle)`;
midpoint heading integration. `vy` is always `0` — the car only moves along its
heading, so it **cannot drift**. Deterministic, with optional Gaussian noise
(`noise_vx`, `noise_yaw_rate`) via `step_stochastic`.
Params: `max_accel, max_speed, drag, wheelbase, max_steer_angle`.

### `arcade` — `ArcadeDriftModel`
Game-like drift model where body-frame `(vx, vy)` is the state. Throttle drives
`vx` (with drag); past `slip_lowerbound` the rear tires spin, shedding drive
force and lateral grip. Steering yaws the car as a kinematic bicycle (scaled by
speed, so it can't turn while parked), but the yaw rate is capped by the front
axle's friction limit — cranking the wheel at speed *understeers* instead of
snapping the car around. The velocity vector is then rotated into the
freshly-yawed body frame — turning faster than the velocity follows is what
creates the sideways slide — and `grip` (friction-capped, slackened by slip)
bleeds that lateral velocity away. Low `grip` → longer, looser slides; high
`grip` ≈ kinematic. Defaults tuned to the real car on a slick floor (µ ≈ 0.4).
Deterministic. Params: `max_speed, max_reverse_speed, max_accel, drag,
wheelbase, max_steer_angle, front_grip_limit, rear_grip_limit,
slip_lowerbound, slip_upperbound, slip_max, slip_decay_rate, slip_accel_loss,
grip, drift_grip`.

### `old_standard` / `old_slip_angle` — predecessor ports (`models/old/`)
Faithful ports of the WS 25/26 project's vehicle models, converted from pixels /
per-frame updates to SI + `dt` and wrapped in the `VehicleModel` interface.
`old_standard` is a kinematic bicycle whose throttle commands a *target speed*
(separate accel/brake/friction rates, reverse supported). `old_slip_angle` is a
scripted arcade drift model: throttle builds `slip`, hard steering at speed
enters a drifting mode (velocity lags the heading), slip angles past 90° trigger
a spin-out. The drift state machine lives *inside* the model instance, so step
one instance sequentially. Details and the list of fixed bugs: `old/README.md`.

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
