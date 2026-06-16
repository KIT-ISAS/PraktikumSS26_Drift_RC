# RL Simulation Refactor — Session Handoff

> **For a new session:** read this file first, then
> [RL_Simulation_Pipeline.md](RL_Simulation_Pipeline.md) (the user's goals/requirements).
> The original implementation plan is at `/root/.claude/plans/i-dont-like-the-tingly-axolotl.md`.
> This doc captures what was built, the conventions/gotchas discovered, how to verify, and the
> remaining open steps so you don't have to re-derive context.

## TL;DR

Refactoring the drift-car simulation + RL stack into **one ROS-free core library**
(`vehicle_simulation`) that owns everything not tied to ROS (track geometry, vehicle models, the Gym
env, data pipeline, training), with **thin ROS wrappers** (`track_pkg`, `simulation_pkg`) that import
from it. **Phase 0 (the refactor) is DONE and verified.** Phases 1–3 (new capabilities) are scoped
but **not started**.

Goal of the whole effort: record manual-drifting rosbags → train a vehicle model of the car → train
an RL agent in sim (no ROS, GPU-friendly) → inspect models/agents offline in Foxglove via MCAP files
→ deploy the agent to the real car. Must support trying multiple models/training approaches,
comparing model-vs-bag and multiple agents, and Xbox-controller sim driving. **No duplicated code
between ROS and training.**

## Decisions already made with the user (do not re-litigate)

1. Deliver the **refactor now + a phased roadmap**.
2. **Track geometry's single source of truth is the ROS-free core lib**; `track_pkg` imports it.
3. **Offline inspection = write MCAP files** (via the `rosbags` dep) opened directly in Foxglove with
   **no ROS running**. (Chosen over a live ROS replay node, for the GPU-server / no-ROS goal.)
4. Observation/action construction is **shared** between the training env and the real-car deployment
   node (sim-to-real guarantee).

## Architecture (target)

```
drift_ws/src/vehicle_simulation/            # ROS-free pip lib (importable on a GPU server, no ROS)
  vehicle_simulation/
    track/        BezierTrack, TrackProjector, load_track(), list_tracks()   # light deps: math/yaml/bisect
    tracks/       figure_8.yaml   (package_data; canonical track data)
    models/       base, kinematic, neural(+DynamicsNet), __init__.build_model()  # torch lazy → neural only
    environments/ drift_env (DriftEnv, lazy), observation.py (shared obs/action builder)
    data/         rosbag_reader, preprocessing
    training/     train_model (neural dynamics), train_agent (SB3 SAC)
  tests/          test_bezier_track.py, test_track_projector.py   # pytest, not colcon
  pyproject.toml  # base deps + extras: [neural]=torch, [rl]=gymnasium+stable-baselines3

drift_ws/src/track_pkg/        # thin ROS wrapper: node imports vehicle_simulation.track
drift_ws/src/simulation_pkg/   # thin ROS wrapper: simulation_node drives a core model (Xbox feel-test)
```

Shared de-duplicated seams: track geometry+data (core `track/`+`tracks/`), obs/action
(`environments/observation.py`), topic names (`shared_pkg.Topics`, used only by ROS wrappers).

## What Phase 0 did (DONE, all verified green)

- **De-duplicated track geometry.** Deleted `track_pkg/track_pkg/bezier_track.py`,
  `track_projector.py`, and `track_pkg/config/figure_8.yaml`. `track_pkg`'s
  `track_publisher_node.py` now does `from vehicle_simulation.track import TrackProjector, load_track`
  and its `track_file` param defaults to the name `"figure_8"` (resolved by `load_track`). Removed the
  config `data_files` entry from `track_pkg/setup.py` and the unused `ament_index_python` dep from its
  `package.xml`. Geometry unit tests moved to `vehicle_simulation/tests/` (imports repointed).
- **Fixed torch coupling.** Added lazy `vehicle_simulation/models/__init__.py::build_model(model_type,
  config=None, weights=None, **kw)`; renamed private `_DynamicsNet` → public `DynamicsNet`; made
  `environments/__init__.py` import `DriftEnv` lazily via module `__getattr__`. `simulation_node` and
  `train_agent` now use `build_model()` + `load_track()`. Result: kinematic sim / track / observation
  all import with **no torch and no gymnasium** installed.
- **Shared obs/action builder.** New `environments/observation.py`
  (`build_observation`, `observation_from_state`, `action_from_array`, `OBS_DIM`, `OBS_LOW`,
  `OBS_HIGH`, `ACTION_DIM`); `DriftEnv` uses it. The future deployment node must reuse it.
- **Install / lint automation.**
  - `vehicle_simulation/pyproject.toml`: base deps `numpy, scipy, pandas, pyyaml, rosbags`; extras
    `[neural]=torch`, `[rl]=gymnasium, stable-baselines3`; `package-data` ships `tracks/*.yaml`.
  - `.devcontainer/scripts/post_start.sh`: editable-installs the core lib (base only) into `/opt/venv`.
  - `.devcontainer/Dockerfile`: adds `/root/repo/drift_ws/src/vehicle_simulation` to `PYTHONPATH`
    (see gotcha below).
  - Registered `vehicle_simulation` as first-party in `pyproject.toml` (Ruff `known-first-party`),
    `.devcontainer/scripts/init_flake8.sh` (`PACKAGES`), and `update_package_list.sh` (appends it
    since `colcon list` can't see a package without `package.xml`).
  - Made `init_flake8.sh` **idempotent** (it previously appended duplicate config keys on every
    container restart → `configparser.DuplicateOptionError` broke `colcon test` for all packages).
  - Added the standard ament linter tests to `simulation_pkg/test/` (it had none).

## Gotchas / conventions discovered (important)

- **PYTHONPATH vs editable install.** This repo *appends* the venv `site-packages` to `PYTHONPATH`
  and runs ROS nodes under **system** `python3` (not the venv python). Pip's editable install uses a
  `.pth`-based meta-path finder that is **only processed by the venv python**, so an editable install
  alone is invisible to ROS nodes. Fix used: put the lib **source dir on `PYTHONPATH`** via the
  Dockerfile (editable-by-nature for both pythons). → **A container rebuild is required** for the
  current dev container's ROS nodes to import `vehicle_simulation`. Until then, `colcon test` and the
  core pytest still pass; only live `ros2 run/launch` of the nodes needs the rebuild.
- **Lazy imports are mandatory** in core `__init__` files: never import `torch`/`gymnasium`/`sb3` at
  package-import time. `models/__init__` and `environments/__init__` follow this — preserve it.
- **Two linters must agree:** Ruff (`pyproject.toml`, line-length 99, double quotes, `D213` style =
  blank first docstring line) and ament_flake8 (`colcon test`). Module docstrings with backslashes
  need `r"""` (D301). `update_package_list.sh` regenerates the first-party lists on container start —
  don't hand-edit those lists except via that script's logic.
- `vehicle_simulation` is **not** a colcon package (no `package.xml`); its tests run via **pytest**,
  not `colcon test`.

## How to verify (copy-paste)

```bash
cd /root/repo
# Lint
ruff check drift_ws/src/vehicle_simulation drift_ws/src/track_pkg drift_ws/src/simulation_pkg
ruff format --check drift_ws/src/vehicle_simulation drift_ws/src/track_pkg drift_ws/src/simulation_pkg
# Core lib tests (system python3 has pytest via ROS; export PYTHONPATH for the src dir)
PYTHONPATH="/root/repo/drift_ws/src/vehicle_simulation:$PYTHONPATH" \
  python3 -m pytest drift_ws/src/vehicle_simulation/tests -q
# Decoupling check (must pass with NO torch / NO gymnasium installed)
PYTHONPATH="/root/repo/drift_ws/src/vehicle_simulation:$PYTHONPATH" python3 -c \
 "from vehicle_simulation.models import build_model, VehicleAction, VehicleState; \
  from vehicle_simulation.track import load_track; \
  build_model('kinematic').step(VehicleState(), VehicleAction(0.2,0.5), 0.02); \
  load_track('figure_8'); print('ok')"
# ROS packages
cd drift_ws && colcon build --packages-select track_pkg simulation_pkg \
  && colcon test --packages-select track_pkg simulation_pkg --event-handlers console_direct+
```

Install heavy extras on a training / GPU machine:
`pip install -e drift_ws/src/vehicle_simulation[neural,rl]`

## OPEN STEPS (not started)

### Operational (one-time, by the user)
- [ ] **Rebuild the dev container** so the Dockerfile `PYTHONPATH` change takes effect and ROS nodes
      can import `vehicle_simulation`. (VS Code will prompt after the `.devcontainer` changes.)
- [ ] Commit Phase 0 (currently unstaged; on `main`, so branch first). Note: `vehicle_simulation`,
      `simulation_pkg`, and the rosbags under `drift_ws/rosbag2_*` are still untracked — decide what to
      commit and ensure `__pycache__` / `*.egg-info` are git-ignored.

### Phase 1 — model fidelity & comparison
- [ ] Add vehicle models beyond kinematic/neural via the `build_model` registry — notably a
      **dynamic bicycle model** parameterised by *measured properties* (mass, `Iz`, wheelbase, tire
      stiffness). This is what "data + measured properties" in the requirements points at. New file
      `vehicle_simulation/models/dynamic.py`, register it in `models/__init__.build_model`.
- [ ] `vehicle_simulation/eval/compare.py`: open-loop roll a model over a bag's recorded actions
      (reuse `data/rosbag_reader.py` + `data/preprocessing.py`), compute trajectory error vs recorded
      pose, output matplotlib metrics **and** an MCAP (recorded vs predicted `/car/pose`) for Foxglove.

### Phase 2 — RL training inspection (no ROS during training)
- [ ] `vehicle_simulation/io/mcap_writer.py`: write poses/track/reward/agent-id to `.mcap` using the
      `rosbags` writer. Topic-name strings passed in as params (defaults documented to match
      `shared_pkg.Topics`) so the core stays ROS-free.
- [ ] Add SB3 `CheckpointCallback` + `EvalCallback` to `training/train_agent.py`; periodically roll
      out the current policy and write an MCAP via the writer.
- [ ] Multi-agent comparison tool: load N checkpoints, roll each out, write one MCAP with namespaced
      pose topics + per-agent markers/reward for side-by-side Foxglove comparison.

### Phase 3 — live ROS viz & deployment
- [ ] Extend `simulation_pkg`'s launch to optionally bring up `track_publisher_node` + `foxglove_bridge`
      so Xbox sim-driving is visible live in Foxglove.
- [ ] Agent **playback** ROS node: load an agent, step a core model, publish `/car/pose` (optionally
      namespaced for live multi-agent compare).
- [ ] **Deployment** node: subscribe `/car/pose` (+ `/track/progress`), build the observation via the
      shared `environments/observation.py`, run the policy, publish `VehicleActionRequest`. Reusing the
      same obs/action builder as training is the sim-to-real guarantee. (Note: on the real car, body-
      frame `vx/vy/yaw_rate` must be estimated from pose — mirror `data/preprocessing.compute_body_frame_velocities`.)

## Key files (quick map)

- Core lib: `drift_ws/src/vehicle_simulation/vehicle_simulation/{track,models,environments,data,training}/`
- Track entry: `track/__init__.py` (`load_track`, `list_tracks`), data in `tracks/figure_8.yaml`
- Model factory: `models/__init__.py` (`build_model`)
- Shared obs: `environments/observation.py`
- ROS wrappers: `track_pkg/track_pkg/track_publisher_node.py`,
  `simulation_pkg/simulation_pkg/simulation_node.py`, `simulation_pkg/launch/simulation.launch.py`
- Automation: `.devcontainer/Dockerfile`, `.devcontainer/scripts/{post_start,init_flake8,update_package_list}.sh`,
  `pyproject.toml`, `drift_ws/src/vehicle_simulation/pyproject.toml`
- Topic registry: `drift_ws/src/shared_pkg/shared_pkg/topics.py` (`Topics`)
- Messages: `drift_ws/src/shared_interfaces/msg/` (`VehicleActionRequest`, `VehicleStatus`, `Track`,
  `TrackProgress`)
