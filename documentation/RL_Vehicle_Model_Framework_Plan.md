# Vehicle-Model Training & Evaluation Framework

> Design of record for the `vehicle_simulation` model training & evaluation
> framework. For package overview, install, and usage see
> [`drift_ws/src/vehicle_simulation/README.md`](../drift_ws/src/vehicle_simulation/README.md).

## Implementation status — ALL PHASES DONE (2026-06-16)

Phases 0–5 are implemented on branch `feat/model-framework` and verified end-to-end
on `rosbag2_2026_06_03-11_43_14`. The plan sections below are the design of record.

- **Phase 0** models/{base,integrate,registry,loader}.py — abstraction + `build_model`.
- **Phase 1** data/{sanity,regimes,dataset,spec}.py — jump-cut + regimes + variable-dt
  multi-step windows + `DatasetSpec`. (Also fixed: custom-msg typestore registration in
  rosbag_reader; `np.split`-on-DataFrame in `segment_episodes`; opt-in accel sanity +
  Savitzky-Golay velocity smoothing for the ~250 Hz Vive data.)
- **Phase 2** models/neural.py rewrite + training/{config,run_dir,trainer,train_model}.py.
- **Phase 3** models/{dynamic,hybrid}.py — physics bicycle + physics-informed hybrid.
- **Phase 4** evaluation/{metrics,mcap_writer,plots,sim,evaluate}.py — 4 eval methods.
- **Phase 5** DataLoader workers + DDP seam (sampler/model wrap behind `cfg.ddp`,
  off by default); `data.preprocessing.build_dataset` deprecated.

**Caveats:** only the 06-03 bag has `/car/pose` (the five 05-29 bags are actions-only).
torch (CPU) is installed in the container but **not** in the Dockerfile (add the
`[neural]`/`[all]` extra there to persist). Method-4 live-sim wiring needs
`simulation_pkg` (PR #52/#53), which is on a parallel branch — reconcile at merge.

### Quickstart

```bash
cd /root/repo
export PYTHONPATH="/root/repo/drift_ws/src/vehicle_simulation:$PYTHONPATH"
# tests (torch tests skip if torch absent)
python3 -m pytest drift_ws/src/vehicle_simulation/tests -q
# train a model (writes a self-contained run dir)
python3 -m vehicle_simulation.training.train_model \
  --bags drift_ws/rosbag2_2026_06_03-11_43_14 --tag v1 \
  --notes "neural, drift+corner+straight"            # --config train.yaml to tune
# evaluate: metrics (overall+per-regime+per-horizon) + recorded-vs-predicted MCAP
python3 -m vehicle_simulation.evaluation.evaluate \
  --run-dir runs/<...> --bags drift_ws/rosbag2_2026_06_03-11_43_14 \
  --mcap-bag drift_ws/rosbag2_2026_06_03-11_43_14
# drive a trained model in sim (needs simulation_pkg):
#   ros2 launch simulation_pkg simulation_demo.launch.py model_type:=runs/<...>
```

Model types: `kinematic` (toy), `dynamic` (physics, usable untrained), `neural`,
`hybrid` (physics + neural residual). `build_model(run_dir)` loads a trained one.

## Current repo state (as of 2026-06-16)

- **Branch when planning:** `feat/car-parameters`. Working tree clean.
- **Open stacked PRs** (created this session, in review — merge bottom-up; GitHub
  auto-retargets each to `main` as the lower one merges):
  - #51 `refactor/sim-core-lib` → `main` — ROS-free core lib + track-geometry de-dup + devcontainer/lint infra + docs. **Base of everything.**
  - #52 `feat/simulation-pkg` → #51 — `simulation_pkg` ROS wrapper (`simulation_node`, kinematic sim, `simulation.launch.py`).
  - #53 `feat/arcade-demo` → #52 — arcade demo model + `keyboard_control` node + `simulation_demo.launch.py`.
  - #54 `feat/car-parameters` → #51 — `cars/drift_rc.yaml` + `car_parameters.py` (`CarParameters`, `load_car_parameters`). **This framework builds on #54.**
- **Base the new framework branch on `feat/car-parameters`** (needs core lib + params).
  Eval **method 4** (live sim) also needs `simulation_pkg` (#52/#53) — do that wiring
  last, ideally after the stack lands on `main`. Keep `models/arcade.py` untouched.
- **Dev container:** already rebuilt, so `PYTHONPATH` now includes the core-lib source
  (`ros2 run/launch` can import `vehicle_simulation`). `~/.config/gh` is mounted.
- **`gh` auth gotcha:** the working token is exported in `~/.bashrc` *after* the
  `[ -z "$PS1" ] && return` guard, so non-interactive tool shells don't get it. Read it
  per command: `export GH_TOKEN=$(grep -oP '^export GH_TOKEN=\K.*' ~/.bashrc | tail -1)`.
  Pushes use SSH (works independently of `gh`). **Token was exposed in a transcript —
  rotate it (`gh auth refresh`) when convenient.**
- **Verify the core lib** (no torch needed for the light path):
  ```bash
  cd /root/repo
  ruff check drift_ws/src/vehicle_simulation && ruff format --check drift_ws/src/vehicle_simulation
  PYTHONPATH="/root/repo/drift_ws/src/vehicle_simulation:$PYTHONPATH" \
    python3 -m pytest drift_ws/src/vehicle_simulation/tests -q
  ```
  `vehicle_simulation` is **not** a colcon package (no `package.xml`); tests run via
  **pytest**, not `colcon test`. Lint: ruff line-length 99, double quotes, D213 (blank
  first line in multi-line docstrings). Torch/gymnasium must stay lazy so the
  kinematic-only path imports without them.

## Reference paper — arXiv 2410.20990 (key findings)

Physics-informed **neural SDE** for autonomous drifting. Dynamic state
`[r (yaw rate), V (speed), β (sideslip), ω_r (rear wheel speed)]`; **accelerations are
NOT states** (only derivatives in the ODE). Inputs `[δ steering angle, τ engine torque]`
in **physical units** — **WE differ:** steering ∈ [−1,1] (nonlinear) and throttle ∈
[−1,1], so those maps must be learned inside our models. Architecture: single-track
bicycle ODE (known mass, a/b CoM, wheel radius, gear) + learned **ExpTanh** tire-force
net + learnable inertias + neural diffusion `Σθ` (the stochastic term). Training:
variable timestep `dt~U(0.01,0.05)`, trajectory segments **0.5–2 s**, Euler–Maruyama,
**NLL loss** (MSE weighted by inverse covariance + log det), Adam lr 1e-3, batch 128,
5 particles. Stochasticity enables zero-shot sim2real without domain randomization. They
report waypoint-tracking error (9–39 cm) but **no per-regime prediction-loss breakdown**
— that's our eval method 2, a gap we add.

## Existing code map (key APIs to reuse — file paths under `drift_ws/src/vehicle_simulation/vehicle_simulation/`)

- `models/base.py`: `VehicleState(x,y,heading,vx,vy,yaw_rate)` (+to/from_array);
  `VehicleAction(steering,throttle)`; `VehicleModel` ABC: abstract `step(state,action,dt)`,
  `step_stochastic` (default=step), classmethod `from_config(path)` (reads YAML `params:`).
- `models/neural.py`: `DynamicsNet(nn.Module)` MLP (in `[vx,vy,yaw_rate,steering,throttle]`,
  out 3 derivatives, optional mean+log_std); `NeuralDynamicsModel(weights_path, stochastic,
  device)` — loads `.pt` with input/output mean/std, Euler-integrates. **Single-step, fixed-dt.**
- `models/kinematic.py`: `KinematicBicycleModel` (numpy, no slip, vy=0). `models/arcade.py`:
  `ArcadeDriftModel` (demo, keep). `models/__init__.py`: `build_model(model_type, config,
  weights, **kw)` lazy factory (kinematic/arcade/neural).
- `data/rosbag_reader.py`: `read_driving_data(bag)->DataFrame[t,x,y,heading,steering,throttle]`
  (reads `/car/pose` PoseStamped + `/car/vehicle_action_request`; yaw from quaternion;
  `merge_asof` backward).
- `data/preprocessing.py`: `compute_body_frame_velocities(df)` (np.gradient + body rotate +
  yaw_rate via unwrap), `filter_stationary(df,min_speed=0.05)`, `segment_episodes(df,
  max_gap_s=1.0)`, `build_dataset(df,dt=0.05)->(states(N,3),actions(N,2),deltas(N,3))`
  (fixed-dt, single-step — **to be replaced by the window dataset**).
- `training/train_model.py`: single-step CLI training of `DynamicsNet` (Adam, MSE/NLL,
  random_split val, saves `.pt`) — **to be rewritten config-driven.**
- `training/train_agent.py`: SB3 SAC on `DriftEnv` via `build_model` + `load_track`.
- `environments/observation.py`: `OBS_DIM=7`, `build_observation`, `observation_from_state`,
  `action_from_array`. `environments/drift_env.py`: Gym env stepping a model (single `_state`).
- `car_parameters.py`: frozen `CarParameters` + `load_car_parameters("drift_rc")`. Derived:
  `tire_radius_m`, `total_gear_reduction`, `front/rear_axle_load_fraction`,
  `com_distance_from_front_axle_m` (=a), `com_distance_from_rear_axle_m` (=b),
  `tracker_offset_from_com_m`. **No Izz, no tire stiffness** in the YAML (see open choice 1).
- Packaged-data + name-vs-path loader pattern to mirror: `track/__init__.py load_track()`
  and `car_parameters.load_car_parameters()`.
- `pyproject.toml`: base deps numpy/scipy/pandas/pyyaml/rosbags; extras `[neural]=torch`,
  `[rl]=gymnasium,stable-baselines3`. `rosbags` can write **MCAP** (StoragePlugin.MCAP,
  `Stores.ROS2_JAZZY` typestore) with no ROS runtime — basis for eval method 3.
- `VehicleActionRequest.msg`: `float32 steering`, `float32 throttle`, `string gyro_mode`,
  `float32 gyro_gain` (only steering/throttle used). Rosbags under `drift_ws/rosbag2_*`
  (gitignored, ~203 MB) contain `/car/pose` + `/car/vehicle_action_request`.

## How to resume (new session)

1. Read this file + the two `documentation/RL_*.md` handoffs.
2. Check whether PRs #51/#52/#54 have merged to `main`; base the framework branch on
   `main` if so, else on `feat/car-parameters`.
3. Start at **Phase 0** below and proceed; the first user-visible win is **Phase 2**
   (first trained run dir reloadable into the sim). Implement each phase as a stacked PR.

## Context

We need to learn a dynamics model of the RC drift car from recorded rosbags, then
use it to train RL drifting agents in simulation. The model must be experimented
with heavily: **multiple model families** (physics, fully-neural, physics-informed
hybrid), each **trainable in different ways** (variable timestep, multi-step rollout,
regime-weighted sampling, deterministic or stochastic), on **differently curated
datasets**, and then **evaluated four ways** (overall loss, per-regime loss,
recorded-vs-predicted Foxglove visualization, and live sim driving with a controller).

The reference is arXiv 2410.20990 (physics-informed neural SDE for drifting:
single-track bicycle core + learned tire model + stochastic diffusion, trained with
variable-dt segment rollouts under an NLL loss). We deviate from it in two ways that
shape the design: we have **normalized steering ∈ [−1,1]** (nonlinearly mapped to a
real angle) instead of a steering angle, and **throttle ∈ [−1,1]** instead of engine
torque — so the steering→angle and throttle→force mappings must be *learned inside the
models*, not inverted in preprocessing. We also have **no measured yaw inertia (Izz)
or tire stiffness**, and the Vive localization occasionally **jumps** (solver error),
which the data pipeline must detect and cut out.

The existing `kinematic`, `neural`, and `arcade` models are placeholders. `arcade`
(a demo model) is kept as-is; `kinematic`/`neural` are cleaned/rewritten. Everything
lives in the ROS-free core lib `drift_ws/src/vehicle_simulation/` so training runs on
a GPU server with no ROS.

Decisions already made with the user: build **models for all approaches**; support
**all training axes**; **file-based run directories** (no W&B/MLflow); **single-GPU
now, DDP-ready** structure; start **deterministic** but include stochastic hooks.

## Branch / PR strategy

This is large — implement in phases, each a stacked PR. Base the work on
`feat/car-parameters` (PR #54: has core lib + `CarParameters`). The framework code is
all core-lib; only eval **method 4** (live sim) touches `simulation_pkg`
(`simulation_demo.launch.py`, PR #52/#53) — do that wiring last, rebasing onto `main`
once #51/#52/#54 land. Keep `models/arcade.py` untouched.

---

## Architecture

### 1. Model abstraction (`models/base.py`, `models/integrate.py`)

A single torch `forward(dyn, action, dt) -> Prediction` is the **source of truth**;
numpy `step()` wraps it for fast inference. New types in `base.py` (keep
`VehicleState`/`VehicleAction` backward-compatible):

- `StateSpec(dynamic_names: tuple[str,...], needs_accel=False, needs_wheel_speed=False)`
  — each model declares the dynamic signals it carries beyond canonical pose.
- `ModelState(pose[x,y,heading], dynamic[D], spec)` with
  `to_vehicle_state()` / `from_vehicle_state(vs, spec)` — internal carrier so models
  needing `accel`/`omega_r`/`beta` don't pollute the canonical `VehicleState`
  (which stays the interchange type for sim/RL/track/dataset).
- `Prediction(mean, cov|None)` — deterministic ⇒ `cov=None`; stochastic ⇒ diagonal/full
  covariance. `step()` uses `mean`; `step_stochastic()` samples.
- `VehicleModel` ABC gains `forward(...)` (torch, optional — only torch models),
  `requires_torch`, `spec`, and classmethods `from_config(path)` (unchanged) +
  `from_run_dir(run_dir, device)`. `step`/`step_stochastic` signatures unchanged so
  `drift_env.py`, `simulation_node.py`, `train_agent.py` keep working.

`models/integrate.py`: `integrate_pose(pose, vx, vy, yaw_rate, dt, method="midpoint")`
and `rollout(model, dyn0, pose0, actions, dts, sample)` — duck-typed for torch **and**
numpy, shared by training and inference. Differentiable rollout is the training
workhorse; `step()` for torch models runs one `forward` under `no_grad` on CPU.

**Default for v1:** keep dynamic state = canonical `(vx, vy, yaw_rate)` so `step()`
stays stateless (safe under `SubprocVecEnv`). Only introduce extra dynamic state
(accel/wheel-speed) for a model that genuinely needs it, and document
single-trajectory-per-instance.

### 2. Concrete models (`models/`)

| Model | File | torch? | Notes |
|-------|------|--------|-------|
| Kinematic (cleaned placeholder) | `kinematic.py` | no | numpy fast RL-bootstrap; keep lazy-import guarantee |
| Arcade (demo) | `arcade.py` | no | **unchanged** |
| Dynamic bicycle (physics) | `dynamic.py` (new) | yes | single-track ODE; known params from `CarParameters` (mass, a/b CoM, tire radius, gear); **learnable** `Cf,Cr`, throttle→Fx gain, steering→δ map, **Izz** (init from box approx `m(L²+W²)/12`) |
| Fully neural (rewrite) | `neural.py` | yes | `DynamicsNet` MLP predicts per-second derivatives (dt-general); `forward` is truth, `step` wraps it; optional `mean+log_std` head |
| Physics-informed hybrid | `hybrid.py` (new) | yes | dynamic-bicycle mean **+ neural residual** on unmodeled tire/force terms; optional neural diffusion `Σθ` ⇒ Neural-SDE when stochastic. The likely end state. |

Steering→angle and throttle→force maps live **inside** the models (learnable), not in
preprocessing — dataset keeps raw normalized actions.

### 3. Data pipeline (`data/`)

Reuse `compute_body_frame_velocities`, `filter_stationary`, `segment_episodes`.
Replace `build_dataset`. New modules:

- `data/sanity.py` — localization-jump detection.
  `detect_discontinuities(df, *, max_speed=8, max_accel=30, max_yaw_rate=25, pos_margin, head_margin)`
  flags transitions where position/heading jump or implied accel exceed
  physically-plausible bounds for the per-sample dt; `split_on_discontinuities(df, mask, min_len)`
  cuts at jumps and keeps clean sub-segments; `sanitize_episodes(...) -> (episodes, JumpReport)`.
- `data/regimes.py` — `label_regimes(df, *, yaw_thresh, vy_thresh, sideslip_thresh, speed_min)`
  adds a `regime` column ∈ {straight, corner, drift, stationary} from
  `yaw_rate`, `vy`, sideslip `β=atan2(vy,vx)`, speed. Thresholds in config (tune from histograms).
- `data/dataset.py` — variable-dt, multi-step **window** dataset.
  `Window(dyn0, actions(H,2), dts(H,), dyn_traj(H+1,D), pose_traj(H+1,3), regime)`;
  `build_windows(episodes, *, horizon, dt_range, stride, resample, seed)`
  (resample each window to dt~U(dt_min,dt_max) for training; keep native dt for eval);
  `WindowDataset(torch Dataset)`; `split_windows(..., by_episode=True)` (episode-level
  split prevents overlapping-window leakage). Single-step = `horizon=1`.
- `data/spec.py` — `DatasetSpec` (bags, car, filters, regimes_included, horizon,
  dt_range, resample, stride, splits, seed, **notes** free-text, plus computed
  `n_windows`, per-regime counts, jump report, created timestamp). `to_yaml/from_yaml`.
  `build_dataset_from_spec(spec) -> (train/val/test WindowDataset, spec_with_stats)` —
  the reproducible "synthesize a dataset differently" entry point (drift-only,
  straight-only, different H, different dt).

### 4. Training framework (`training/`)

- `training/config.py` — `TrainConfig` dataclass (YAML): `model_type`, `model` kwargs,
  embedded `DatasetSpec`, `horizon`, `loss` (mse|nll), `rollout_loss`, `epochs`,
  `batch_size`, `lr`, `weight_decay`, `optimizer`, `scheduler`, `regime_weights`,
  `n_particles`, `val_every`, `early_stop_patience/min_delta`, `device="auto"`,
  `ddp=False`, `seed`, `grad_clip`.
- `training/trainer.py` — config-driven `Trainer`. Device-agnostic; **DDP seams**
  (all moves via `self.device`; sampler swappable to `DistributedSampler`, model
  wrappable in `DistributedDataParallel` behind `cfg.ddp` — off in v1).
  Regime-weighted `WeightedRandomSampler`. Loss via `rollout(...)` over `horizon`:
  MSE (deterministic) or NLL `0.5(((y−μ)ᵀΣ⁻¹(y−μ)) + logdet Σ)` summed over steps
  (Euler–Maruyama, `n_particles`). Validation every `val_every` (+ per-regime val
  loss); early stop on patience; save best. Writes run-dir artifacts incrementally.
- `training/run_dir.py` — run directory layout:
  ```
  runs/<timestamp>__<model_type>__<tag>/
    config.yaml          dataset_spec.yaml      notes.md
    model.pt             model_last.pt          metrics.csv
    loss_curve.png       meta.json (git sha, host, device, versions, seed, wall time)
    eval/                (metrics.json, mcap/, plots — filled by eval suite)
  ```
  `model.pt` is **self-contained**: `{state_dict, model_type, StateSpec,
  normalization, init_args}` so it reloads with no external config.
- `models/loader.py` — `load_model_from_run(run_dir, device, stochastic) -> VehicleModel`
  (single reload entry point for eval, `build_model`, and sim).
- `training/train_model.py` — **rewrite** the CLI: `--config config.yaml [--tag] [--notes]`
  + a few overrides (`--epochs`, `--device`); builds dataset from spec, runs `Trainer`.

### 5. Evaluation suite (`evaluation/`)

- `evaluation/metrics.py` (methods 1 & 2): `evaluate_model(model, dataset, *, horizon)
  -> EvalReport{overall, per_regime, per_horizon}` — single-step + multi-step rollout
  error + pose RMSE + NLL; `per_regime` runs on single-regime test subsets ("where it
  works/fails"); `per_horizon` is the error-growth curve. `to_json/save`.
- `evaluation/mcap_writer.py` (method 3): `write_eval_mcap(out, recorded_df,
  predicted_traj, *, topics_prefix="/eval")` via `rosbags` `Writer`
  (`StoragePlugin.MCAP`, `Stores.ROS2_JAZZY` typestore — no ROS runtime). Topics:
  `/eval/pose_recorded`, `/eval/pose_predicted` (PoseStamped), `/eval/path_recorded`,
  `/eval/path_predicted` (nav_msgs/Path overlay), `/eval/error_lateral`,
  `/eval/error_heading`, `/eval/cmd_steering`, `/eval/cmd_throttle` (Float64).
  Helper `rollout_from_recording(model, df, start_idx, horizon)` = open-loop prediction
  from a real recording's initial state + recorded actions.
- `evaluation/plots.py`: `plot_loss_curve`, `plot_per_horizon`, `plot_regime_bars`
  (matplotlib → new `eval` extra).
- `evaluation/sim.py` (method 4): `load_for_sim(run_dir, device="cpu")`. No node
  rewrite — extend `build_model` to resolve a run dir; drive via existing
  `simulation_demo.launch.py model_type:=<run_dir>` + `keyboard_control`/`joystick:=true`.

### 6. Registry (`models/__init__.py`, `models/registry.py`)

`MODEL_REGISTRY = {kinematic, dynamic, neural, hybrid}` (lazy class getters; arcade
stays its own entry). `build_model(model_type, *, config=None, weights=None,
run_dir=None, device="cpu", **kw)` resolution (mirrors `load_track`/`load_car_parameters`):
1. `run_dir` given, or `model_type` is a dir containing `model.pt` ⇒ `load_model_from_run`;
2. registry name ⇒ construct;
3. kinematic ⇒ `from_config`/defaults. Back-compat: `build_model("neural", weights="x.pt")`
   and `build_model("kinematic")` still work.

### Dependencies (`pyproject.toml`)

`neural=["torch"]` already covers dynamic/neural/hybrid. Add `eval=["matplotlib"]`.
MCAP needs only `rosbags` (base). Optional `all=[neural+rl+eval]`. Ship `cars/*.yaml`
already done; no new package-data.

---

## Phased implementation (vertical slice first)

- **Phase 0 — Abstraction + registry.** `base.py` (StateSpec/ModelState/Prediction/ABC,
  back-compat), `models/integrate.py`, cleaned `kinematic.py`, registry + run-dir
  resolution in `build_model`. Verifiable in the existing sim immediately.
- **Phase 1 — Data pipeline.** `data/{sanity,regimes,dataset,spec}.py`. Torch-free,
  testable on synthetic DataFrames. Unblocks everything.
- **Phase 2 — Neural rewrite + Trainer (FIRST FULL SLICE).** `neural.py` rewrite,
  `training/{config,run_dir,trainer}.py` (start `horizon=1` MSE → add rollout → NLL →
  regime-weighted), `models/loader.py`, `train_model.py` CLI. Produces the first run dir:
  rosbag → spec → windows → train → run dir → reload → sim.
- **Phase 3 — Physics + hybrid.** `models/dynamic.py`, `models/hybrid.py` through the
  same Trainer.
- **Phase 4 — Evaluation suite.** `evaluation/{metrics,mcap_writer,plots,sim}.py`,
  `eval` extra, launch wiring for method 4.
- **Phase 5 — DDP seams + polish.** `DistributedSampler`/DDP behind `cfg.ddp`,
  deprecate `build_dataset`, docs, update the handoff doc.

---

## Verification

New pytest under `tests/` (torch tests guarded `pytest.importorskip("torch")`):
- `test_model_abstraction.py` — `ModelState` round-trip; kinematic `step` with no torch.
- `test_sanity.py` — injected position/heading jump flagged exactly; clean data flags nothing.
- `test_regimes.py` — straight/corner/drift labels at threshold boundaries.
- `test_dataset.py` — window shapes, dt within range, `by_episode` split has no leakage,
  `DatasetSpec` YAML round-trip.
- `test_train_smoke.py` — tiny synthetic linear dynamics; Trainer 2–3 epochs at
  `horizon=1` then `>1`, loss decreases, run-dir artifacts present, NLL path runs.
- `test_run_dir_roundtrip.py` — train → `load_model_from_run` → `step()` reproduces;
  `build_model(run_dir=...)` works.
- `test_mcap_writer.py` — write then reopen with `rosbags` Reader; expected topics/counts (no torch).
- `test_eval_metrics.py` — `EvalReport` has overall/per_regime/per_horizon; per-horizon error grows.
- `test_build_model.py` — registry + run-dir resolution + back-compat.
- Manual: load a run-dir model into `DriftEnv` (method-4 plumbing, no ROS), then full
  Foxglove + keyboard drive as a documented manual check.

Lint: ruff line-length 99, double quotes, D213; keep torch out of import paths reachable
by the kinematic-only sim.

---

## Open design choices (recommended defaults; flag if you disagree)

1. **Izz / tire stiffness unknown** ⇒ make `Izz, Cf, Cr, throttle→Fx, steering→δ`
   **learnable** `nn.Parameter`s in `dynamic.py`, init from box approx + rough guesses;
   optional `yaw_inertia_kgm2` YAML prior. Hybrid residual absorbs the rest.
2. **Keep raw normalized actions** in the dataset; models learn the steering/throttle maps.
3. **Canonical `(vx,vy,yaw_rate)` dynamic state** for v1 models (stateless `step`);
   add extra dynamic state only when a model needs it.
4. **MCAP** uses standard msg types (PoseStamped/Path/Float64); emit actions as Float64
   (custom `VehicleActionRequest` isn't in the bundled typestore).
5. **Split by episode** by default.
6. **Self-contained checkpoints** carry `model_type`/`StateSpec` metadata; prefer a torch
   safe-globals allowlist over blanket `weights_only=False`.

---

## Ideas / experiment menu (the "give ideas" ask)

**Model types to try**
- Kinematic bicycle (baseline / RL bootstrap, no slip).
- Linear dynamic bicycle (linear tires) — cheap, valid for mild slip.
- Nonlinear dynamic bicycle (Pacejka/Fiala or learned ExpTanh tire) — captures drift.
- Fully-neural MLP on `[vx,vy,yaw_rate,steering,throttle]` → derivatives.
- Physics-informed hybrid: bicycle + neural residual (recommended end state).
- Neural-SDE (hybrid + learned diffusion `Σθ`) for stochastic / uncertainty-aware sim2real.
- Sequence models (GRU/Temporal-Conv) if first-order Markov state proves insufficient
  (e.g. unmodeled actuator lag) — add history to the input rather than the state.

**Training procedures**
- Single-step derivative regression (fast sanity baseline).
- Multi-step rollout loss (compounding-error robustness) — sweep horizon H.
- Variable-dt sampling (dt~U(0.01,0.05)) for dt-general models.
- Stochastic NLL with N particles (uncertainty; avoids domain randomization).
- Curriculum: straight → corner → drift, or short→long horizons.
- Regime-weighted / regime-balanced sampling (up-weight rare drift windows).
- Pose-space vs state-space loss weighting; Jacobian/spectral regularization for stable
  rollouts; teacher-forcing ratio scheduling.

**Data curation methods**
- Localization-jump sanity cutting (the solver-error fix) with a saved jump report.
- Regime-specific datasets (drift-only, straight-only) to probe specialization.
- Segment-length and dt sweeps (short reactive vs long-horizon datasets).
- Per-tire / per-surface / per-session splits to test generalization.
- Stationary/low-speed filtering; outlier velocity/accel clipping.
- Balanced regime mixes; train/val/test split **by episode/session** to avoid leakage.
- Free-text `notes.md` per dataset describing exactly what was curated and why.
