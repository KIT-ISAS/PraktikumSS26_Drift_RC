# Training the drift agent on a GPU server

The SAC drift agent trains in pure simulation (no ROS, no car), so it runs on any
GPU box. A run is a self-contained `runs/<timestamp>__sac__<tag>/` directory
([`run_dir.py`](run_dir.py)) — train remotely, then `scp` the run dir back.

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

Base `vehicle_simulation` is lightweight; training needs the `[all]` extra
(torch + gymnasium + stable-baselines3 + matplotlib). Install into the venv:

```bash
pip install -e "/root/repo/drift_ws/src/vehicle_simulation[all]"
```

## 4. Train the curriculum

Three stages, each warm-started from the previous run. Training auto-selects CUDA
when available; confirm via the `device` field in the run's `meta.json`.

```bash
cd /root/repo/drift_ws/src/vehicle_simulation

/opt/venv/bin/python -m vehicle_simulation.agents.train_agent --config stage1_path_follow --tag s1 > runs/s1_train.log 2>&1
/opt/venv/bin/python -m vehicle_simulation.agents.train_agent --config stage2_fast  --init-from runs/<stage1_run> --tag s2 > runs/s2_train.log 2>&1
/opt/venv/bin/python -m vehicle_simulation.agents.train_agent --config stage3_drift --init-from runs/<s3> --tag s3 > runs/s3_train.log 2>&1
```

Useful flags: `--total-steps N` (override), `--seed N`, `--runs-dir <path>`.
Each run writes `config.yaml`, `meta.json`, `progress.csv`, `agent.zip`, and
`best_model.zip` (plus `eval/`). Optionally evaluate on the server:

```bash
python -m vehicle_simulation.agents.evaluate_agent --run-dir runs/<stage3_run> --episodes 10
```

## 5. Sync runs back with rsync

Run dirs are self-contained and effectively write-once, so an incremental sync only
transfers new/changed runs. From your **local machine** (the container mounts the
repo at `/root/repo`, so runs live under the host clone path):

```bash
rsync -avz --progress \
  user@gpu-server:<repo>/drift_ws/src/vehicle_simulation/runs/ \
  ./drift_ws/src/vehicle_simulation/runs/
```

Trailing slashes matter: `runs/` on both sides copies contents into local `runs/`
(without them you get a nested `runs/runs/`). Add `--delete` to make local an exact
mirror, or `--exclude='*.log'` to skip the `*_train.log` files.

## Fine-tuning on real-car data

Record real transitions and train offline afterwards.

**1. Record on the car** Drive the sim policy with

```bash
ros2 launch agent_pkg record.launch.py run_dir:=/path/to/runs/<sim_run>
# ros2 topic pub --once /agent/start std_msgs/msg/Empty {}
```

Transitions land in the node's `output_dir` (default `recordings/`) as
`<timestamp>__recording.npz`.

**2. Sync the recordings** to GPU server.

**3. Fine-tune off-board.**

```bash
/opt/venv/bin/python -m vehicle_simulation.agents.finetune_agent \
    --init-from runs/<sim_run> --data recordings/*.npz --tag real-s3 --gradient-steps 20000
```

Useful flags: `--batch-size`, `--learning-rate` (keep modest — offline SAC can over-estimate),
`--sim-rollout-steps N` (interleave a little sim experience as a regularizer), `--device`,
`--config` (override the stage).

## Local CPU training

No GPU needed for a quick smoke run or a short curriculum — the same commands work
on the local dev container, just slower. Install the CPU-only torch stack (`[rl-cpu]`
pins the `+cpu` wheels, avoiding the multi-GB CUDA download):

```bash
pip install -e "/root/repo/drift_ws/src/vehicle_simulation[rl-cpu]" \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

Then train as in step 4; the run's `meta.json` `device` field will read `cpu`. Keep
runs short with `--total-steps` since CPU rollout is slow:

```bash
cd /root/repo/drift_ws/src/vehicle_simulation
/opt/venv/bin/python -m vehicle_simulation.agents.train_agent --config stage1_path_follow --tag s1-cpu
```
