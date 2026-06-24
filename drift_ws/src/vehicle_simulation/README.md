# vehicle_simulation

ROS-free Python library that is the single source of truth for everything in the
drift-car simulation + reinforcement-learning pipeline that must **not** depend on
ROS: track geometry, vehicle dynamics models, the data pipeline, model training,
and evaluation. The ROS packages (`track_pkg`, `simulation_pkg`) are thin wrappers
that import from here, so the same code runs in ROS nodes and in a no-ROS training
loop — with no duplication.

## Why a separate library (the "second Python environment")

Training a vehicle model or an RL agent should not require a ROS install (e.g. on a
GPU server), and the heavy ML deps should not leak into the ROS nodes. So this lib:

- is a plain pip package, **not** a colcon package (no `package.xml`); its tests run
  via **pytest**, not `colcon test`;
- is editable-installed into the venv at `/opt/venv` and put on `PYTHONPATH`, so both
  the system Python (ROS nodes) and the venv Python can import it
  (see `documentation/Python.md`);
- **lazily imports** heavy optional deps (torch / gymnasium) so the lightweight path
  (track geometry, kinematic/arcade models, data) imports without them.

## Install

```bash
pip install -e drift_ws/src/vehicle_simulation            # base: track geometry + data
pip install -e drift_ws/src/vehicle_simulation[neural]    # + torch models & training
pip install -e drift_ws/src/vehicle_simulation[rl]        # + gymnasium/SB3 (RL, scaffold)
pip install -e drift_ws/src/vehicle_simulation[eval]      # + matplotlib (eval plots)
pip install -e drift_ws/src/vehicle_simulation[all]
```

## Package layout

```
vehicle_simulation/
  track/         Bézier track geometry + projection; load_track(), list_tracks()
  tracks/        packaged track data (figure_8.yaml)
  models/        dynamics models + build_model() factory:
                   kinematic, arcade  (basic, torch-free)
                   neural, dynamic (physics), hybrid (physics + neural residual)
  cars/          measured car parameters (drift_rc.yaml)
  car_parameters.py   load_car_parameters() + derived quantities (CoG, mass, gearing, ...)
  data/          rosbag reader, preprocessing, localization-jump sanity checks,
                 regime labelling, variable-dt multi-step window datasets, DatasetSpec
  training/      config-driven Trainer, file-based run dirs, train_model CLI
  evaluation/    metrics (overall / per-regime / per-horizon), recorded-vs-predicted
                 MCAP for Foxglove, plots, evaluate CLI
  environments/  shared observation/action builder (sim-to-real seam) + RL env scaffold
```

> Components are introduced incrementally across the feature PRs; the layout above
> is the full package.

## Usage

```python
from vehicle_simulation.track import load_track
track = load_track("figure_8")                      # packaged name or a YAML path

from vehicle_simulation.models import build_model
model = build_model("arcade")                        # kinematic | dynamic | a run dir
next_state = model.step(state, action, dt)

from vehicle_simulation.car_parameters import load_car_parameters
car = load_car_parameters("drift_rc")                # measured params + derived props
```

Train and evaluate a dynamics model (needs the `[neural]` / `[eval]` extras):

```bash
python -m vehicle_simulation.training.train_model \
    --bags drift_ws/rosbag2_<...> --tag v1            # -> runs/<id>/ (self-contained)
python -m vehicle_simulation.evaluation.evaluate \
    --run-dir runs/<id> --bags drift_ws/rosbag2_<...> --mcap-bag drift_ws/rosbag2_<...>
```

A trained model is reloaded by passing its run directory to `build_model(run_dir)`,
including into the ROS simulator (`simulation_pkg`).

## Used by the ROS packages

- `track_pkg` imports `vehicle_simulation.track` (geometry lives here, not in the node).
- `simulation_pkg` builds a model with `build_model(...)` and steps it to publish
  `/car/pose` — so any model (basic, trained, physics, hybrid) can be driven in sim.

Keep this logic in the library; do not duplicate it in the ROS packages.

## Testing

```bash
PYTHONPATH="$(pwd)/drift_ws/src/vehicle_simulation:$PYTHONPATH" \
    python3 -m pytest drift_ws/src/vehicle_simulation/tests -q
```

torch-dependent tests skip automatically when torch is not installed.

## Conventions

- Lazy-import heavy deps in package `__init__`s (keep the light path torch-free).
- Lint: Ruff line-length 99, double quotes, `D213` docstrings (must agree with
  ament_flake8 — see `documentation/Ruff.md`).
