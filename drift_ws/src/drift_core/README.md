# drift_core

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
pip install -e drift_ws/src/drift_core            # base: track geometry + data
pip install -e drift_ws/src/drift_core[neural]    # + torch models & training
pip install -e drift_ws/src/drift_core[rl]        # + gymnasium/SB3 (RL, scaffold)
pip install -e drift_ws/src/drift_core[eval]      # + matplotlib (eval plots)
pip install -e drift_ws/src/drift_core[all]

# CPU-only inference on a device with no GPU (deploy/robot): run a trained agent
# without the multi-GB CUDA torch. Pinned to the training checkpoints; the +cpu
# torch wheel needs the PyTorch CPU index:
pip install -e "drift_ws/src/drift_core[rl-cpu]" \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

## Package layout

Three domains — the car, the track it drives on, and the agent that drives it —
plus two shared leaf modules that all three (and the ROS packages) depend on.

```
drift_core/
  transforms.py  quaternion / angle maths, ROS-free (used by 5 ROS packages)
  paths.py       default run-directory locations (drift_ws/runs/agent | vehicle_model)

  track/         Bézier track geometry + projection; load_track(), list_tracks()
    tracks/        packaged track data (figure_8.yaml)

  vehicle_model/  build_model() factory + the VehicleModel interface, re-exported
    models/        dynamics models:
                     kinematic, arcade  (basic, torch-free)
                     neural, dynamic (physics), hybrid (physics + neural residual)
    car_parameters.py  load_car_parameters() + derived quantities (CoG, mass, gearing, ...)
    cars/          measured car parameters (drift_rc.yaml)
    data/          rosbag reader, preprocessing, localization-jump sanity checks,
                   regime labelling, variable-dt multi-step window datasets, DatasetSpec
    training/      config-driven Trainer, file-based run dirs, train_model CLI
    evaluation/    metrics (overall / per-regime / per-horizon), recorded-vs-predicted
                   MCAP for Foxglove, plots, evaluate CLI

  agent/         DriftAgent inference + train/finetune/evaluate/analyze CLIs
    environments/  shared observation/action builder (sim-to-real seam) + DriftEnv
      configs/       packaged curriculum stages (stage1_path_follow, ...)
    control/       deterministic path following: lateral MPC, speed profile, throttle PI
```

## Usage

```python
from drift_core.track import load_track
track = load_track("figure_8")                      # packaged name or a YAML path

from drift_core.vehicle_model import build_model
model = build_model("arcade")                        # kinematic | dynamic | a run dir
next_state = model.step(state, action, dt)

from drift_core.vehicle_model.car_parameters import load_car_parameters
car = load_car_parameters("drift_rc")                # measured params + derived props
```

### Where runs are written

Both trainers write self-contained run directories under `drift_ws/runs/`, split by
what was trained:

```
drift_ws/runs/
  agent/           RL policies      (train_agent, finetune_agent, analyze_runs)
  vehicle_model/   dynamics models  (train_model)
```

These defaults are anchored to the repo checkout (`drift_core/paths.py`), not the
working directory, so runs land in the same place whatever you `cd` into. Override
per command with `--runs-dir`, or globally with `DRIFT_RUNS_DIR` (e.g. a scratch
disk on a GPU server). The tree is git-ignored — share run dirs out-of-band.

Train and evaluate a dynamics model (needs the `[neural]` / `[eval]` extras):

```bash
python -m drift_core.vehicle_model.training.train_model \
    --bags drift_ws/bags/rosbag2_<...> --tag v1   # -> drift_ws/runs/vehicle_model/<id>/
python -m drift_core.vehicle_model.evaluation.evaluate \
    --run-dir drift_ws/runs/vehicle_model/<id> \
    --bags drift_ws/bags/rosbag2_<...> --mcap-bag
```

A trained model is reloaded by passing its run directory to `build_model(run_dir)`,
including into the ROS simulator (`simulation_pkg`) — which is how you drive a
trained model yourself, by keyboard or gamepad:

```bash
ros2 launch simulation_pkg simulation_demo.launch.py \
    model_type:=drift_ws/runs/vehicle_model/<id>
ros2 run joystick_pkg keyboard_control     # separate terminal (or joystick:=true above)
```

Connect Foxglove to `ws://localhost:8765` to watch it.

## RL agent (needs the `[rl]` extra)

The drift agent is a Stable-Baselines3 **SAC** policy trained in `DriftEnv`
(`agent/environments/drift_env.py`), which steps any `build_model(...)` dynamics model
(arcade baseline now, a trained model later) on a track at 50 Hz. Training follows
a three-stage curriculum — each stage is a packaged YAML config
(`agent/environments/configs/`), and later stages warm-start from the previous run:

```bash
python -m drift_core.agent.train_agent --config stage1_path_follow --tag s1
python -m drift_core.agent.train_agent --config stage2_fast  --init-from <s1_run_dir> --tag s2
python -m drift_core.agent.train_agent --config stage3_drift --init-from <s2_run_dir> --tag s3

python -m drift_core.agent.evaluate_agent --run-dir drift_ws/runs/agent/<id> --episodes 10
```

Each run writes a self-contained `drift_ws/runs/agent/<timestamp>__sac__<tag>/` (config, meta,
`progress.csv`, `agent.zip` + `best_model.zip`, `eval/`). Inference — in the ROS
node or anywhere else — goes through `drift_core.agent.DriftAgent`:

```python
from drift_core.agent import DriftAgent
agent = DriftAgent("drift_ws/runs/agent/<id>")   # loads best_model.zip / agent.zip
action = agent.act(obs)                      # -> VehicleAction, deterministic
```

**Sim-to-real seam:** `agent/environments/observation.py` is the single source of truth
for the policy input (local state + upcoming-centerline curvature, fixed
normalization constants, no learned statistics). The deployment ROS node must
build observations with exactly these helpers — every input maps 1:1 to
`/car/odom`, `TrackProgress`, `Track`, and the node's own last published action.
(Follow-up: export the actor to TorchScript if SB3-free deployment is needed.)

## Used by the ROS packages

- `track_pkg` imports `drift_core.track` (geometry lives here, not in the node).
- `simulation_pkg` builds a model with `build_model(...)` and steps it to publish
  `/car/pose` — so any model (basic, trained, physics, hybrid) can be driven in sim.
- `agent_pkg` runs a trained agent via `DriftAgent` and builds its observations with
  `agent/environments/observation.py` — the same code the agent was trained on.

Keep this logic in the library; do not duplicate it in the ROS packages.

## Testing

```bash
PYTHONPATH="$(pwd)/drift_ws/src/drift_core:$PYTHONPATH" \
    python3 -m pytest drift_ws/src/drift_core/tests -q
```

torch-dependent tests skip automatically when torch is not installed.

## Conventions

- Lazy-import heavy deps in package `__init__`s (keep the light path torch-free).
- Lint: Ruff line-length 99, double quotes, `D213` docstrings (must agree with
  ament_flake8 — see `documentation/Ruff.md`).
