# Training the drift agent on a GPU server

The SAC drift agent trains in pure simulation (no ROS, no car), so it runs on any
GPU box. A run is a self-contained `drift_ws/runs/agent/<timestamp>__sac__<tag>/`
directory ([`run_dir.py`](run_dir.py)) — train remotely, then `scp` the run dir back.

## 1. Connect and get the code

```bash
ssh user@gpu-server
git clone <repo-url> && cd <repo>          # or: git pull in an existing clone
git checkout <branch>
```

## 2. Start the dev container with GPU

The host needs an NVIDIA GPU with `nvidia-container-toolkit` installed and the
Docker daemon configured for the `nvidia` runtime. Layer the GPU overlay on top of
the base compose file (see [`docker-compose.gpu.yaml`](../../../../../.devcontainer/docker-compose.gpu.yaml)):

```bash
cd .devcontainer
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d --build
docker exec -it ros2-image bash            # container name may vary; check `docker compose ps`
```

Verify the GPU is visible inside the container with `nvidia-smi`.

## 3. Install the training extras

Base `drift_core` is lightweight; training needs the `[all]` extra
(torch + gymnasium + stable-baselines3 + matplotlib). Install into the venv:

```bash
pip install -e "/root/repo/drift_ws/src/drift_core[all]"
```

## 4. Train the curriculum

Three stages, each warm-started from the previous run. Training auto-selects CUDA
when available; confirm via the `device` field in the run's `meta.json`.

Runs are written to `drift_ws/runs/agent/` regardless of the working directory
(see [`paths.py`](../paths.py)), so you can launch from anywhere in the repo:

```bash
cd /root/repo
RUNS=drift_ws/runs/agent

/opt/venv/bin/python -m drift_core.agent.train_agent --config stage1_path_follow --tag s1 > $RUNS/s1_train.log 2>&1
/opt/venv/bin/python -m drift_core.agent.train_agent --config stage2_fast  --init-from $RUNS/<stage1_run> --tag s2 > $RUNS/s2_train.log 2>&1
/opt/venv/bin/python -m drift_core.agent.train_agent --config stage3_drift --init-from $RUNS/<stage2_run> --tag s3 > $RUNS/s3_train.log 2>&1
```

Useful flags: `--total-steps N` (override), `--seed N`, `--runs-dir <path>`
(or `DRIFT_RUNS_DIR=<path>` to relocate the whole tree, e.g. onto a scratch disk).
Each run writes `config.yaml`, `meta.json`, `progress.csv`, `agent.zip`, and
`best_model.zip` (plus `eval/`). Optionally evaluate on the server:

```bash
python -m drift_core.agent.evaluate_agent --run-dir $RUNS/<stage3_run> --episodes 10
```

## 5. Sync runs back with rsync

Run dirs are self-contained and effectively write-once, so an incremental sync only
transfers new/changed runs. From your **local machine** (the container mounts the
repo at `/root/repo`, so runs live under the host clone path):

```bash
rsync -avz --progress \
  user@gpu-server:<repo>/drift_ws/runs/agent/ \
  ./drift_ws/runs/agent/
```

Trailing slashes matter: `agent/` on both sides copies contents into the local
`agent/` (without them you get a nested `agent/agent/`). Add `--delete` to make local
an exact mirror, or `--exclude='*.log'` to skip the `*_train.log` files.

## Fine-tuning on real-car data

Record real transitions and train offline afterwards.

**1. Record on the car** Drive the sim policy with

```bash
ros2 launch agent_pkg record.launch.py run_dir:=drift_ws/runs/agent/<sim_run>
# ros2 topic pub --once /agent/start std_msgs/msg/Empty {}
```

Transitions land in the node's `output_dir` (default `recordings/`) as
`<timestamp>__recording.npz`.

**2. Sync the recordings** to GPU server.

**3. Fine-tune off-board.**

```bash
/opt/venv/bin/python -m drift_core.agent.finetune_agent \
    --init-from $RUNS/<sim_run> --data recordings/*.npz --tag real-s3 --gradient-steps 20000
```

Useful flags: `--batch-size`, `--learning-rate` (keep modest — offline SAC can over-estimate),
`--sim-rollout-steps N` (interleave a little sim experience as a regularizer), `--device`,
`--config` (override the stage).

## Learning from human demonstrations

Same offline pipeline, but a human drives: `agent_pkg`'s `demo_recorder_node`
passively reconstructs training-format transitions from the joystick's
`VehicleActionRequest` commands (SAC is off-policy, so human transitions are valid
replay data). No `run_dir`/policy is needed — a `stage_config` supplies the
reward/termination config, and it must match the stage you fine-tune on.

**1. Record while driving manually.** The full-stack launch also captures a raw
rosbag (`~/recordings/human/bags/`) so observations can be recomputed offline if
the observation format ever changes:

```bash
ros2 launch drift_ws/launch/human_demo_recording.launch.py stage_config:=stage3_drift
```

Recording starts automatically once the car is on track and moving; episodes end on
off-track/stall exactly like the training env, and recording holds until the car has
recovered. Files land in `~/recordings/human/` as `<timestamp>__human_demo.npz`.

**Drive with gyro gain 0** (d-pad down; the manual-driving config defaults to 1.0):
the ESP32's gyro assist corrects the steering *after* it is logged, and the training
sim has no such assist, so assisted demos teach dynamics the agent will never see.
The recorder latches the session's `gyro_mode`/`gyro_gain` into the npz metadata,
warns if the gain is nonzero, and holds recording if the settings change mid-session
until they are restored.

**2. Fine-tune off-board** exactly as above, pointing `--data` at the demo files:

```bash
/opt/venv/bin/python -m drift_core.agent.finetune_agent \
    --init-from $RUNS/<sim_run> --data recordings/*__human_demo.npz --tag human-s3
```

Human demos are off-distribution for the critic; mixing them with the policy's own
recordings and/or `--sim-rollout-steps` keeps offline SAC better behaved.

## Local CPU training

No GPU needed for a quick smoke run or a short curriculum — the same commands work
on the local dev container, just slower. Install the CPU-only torch stack (`[rl-cpu]`
pins the `+cpu` wheels, avoiding the multi-GB CUDA download):

```bash
pip install -e "/root/repo/drift_ws/src/drift_core[rl-cpu]" \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

Then train as in step 4; the run's `meta.json` `device` field will read `cpu`. Keep
runs short with `--total-steps` since CPU rollout is slow:

```bash
/opt/venv/bin/python -m drift_core.agent.train_agent --config stage1_path_follow --tag s1-cpu
```
