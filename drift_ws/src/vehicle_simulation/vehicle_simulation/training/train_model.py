r"""
Train a NeuralDynamicsModel from rosbag driving data.

Usage:
    python -m vehicle_simulation.training.train_model \
        --bags path/to/bag1 path/to/bag2 \
        --out weights/neural_model.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from ..data.preprocessing import (
    build_dataset,
    compute_body_frame_velocities,
    filter_stationary,
    segment_episodes,
)
from ..data.rosbag_reader import read_driving_data
from ..models.neural import DynamicsNet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train neural dynamics model from rosbag data")
    p.add_argument("--bags", nargs="+", required=True, help="Paths to rosbag2 bag directories")
    p.add_argument("--out", required=True, help="Output .pt checkpoint path")
    p.add_argument("--dt", type=float, default=0.05, help="Resampling timestep (s)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Train stochastic model (NLL loss, outputs mean+log_std)",
    )
    p.add_argument("--val-fraction", type=float, default=0.1)
    return p.parse_args()


def _load_all_bags(bag_paths: list[str], dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_states, all_actions, all_deltas = [], [], []
    for bag_path in bag_paths:
        print(f"Loading {bag_path} …")
        try:
            df = read_driving_data(bag_path)
            df = compute_body_frame_velocities(df)
            df = filter_stationary(df)
            episodes = segment_episodes(df)
            for ep in episodes:
                s, a, d = build_dataset(ep, dt=dt)
                if len(s) > 0:
                    all_states.append(s)
                    all_actions.append(a)
                    all_deltas.append(d)
        except Exception as exc:
            print(f"  Warning: skipping {bag_path}: {exc}")

    if not all_states:
        raise RuntimeError("No training data loaded from any bag")

    return (
        np.concatenate(all_states),
        np.concatenate(all_actions),
        np.concatenate(all_deltas),
    )


def _train_epoch(model, loader, optimizer, stochastic: bool) -> float:
    model.train()
    total = 0.0
    for x_batch, y_batch in loader:
        pred = model(x_batch)
        loss = _compute_loss(pred, y_batch, stochastic, model.output_dim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x_batch)
    return total / len(loader.dataset)


@torch.no_grad()
def _val_epoch(model, loader, stochastic: bool) -> float:
    model.eval()
    total = 0.0
    for x_batch, y_batch in loader:
        pred = model(x_batch)
        total += _compute_loss(pred, y_batch, stochastic, model.output_dim).item() * len(x_batch)
    return total / len(loader.dataset)


def _compute_loss(
    pred: torch.Tensor, target: torch.Tensor, stochastic: bool, n: int
) -> torch.Tensor:
    if stochastic:
        mu, log_std = pred[:, :n], pred[:, n:]
        std = torch.exp(log_std.clamp(-4.0, 2.0))
        return (((target - mu) / std) ** 2 + 2.0 * log_std).mean()
    return nn.functional.mse_loss(pred, target)


def main() -> None:
    args = _parse_args()

    states, actions, deltas = _load_all_bags(args.bags, args.dt)
    print(f"Total samples: {len(states)}")

    # Normalize inputs and outputs
    inputs = np.concatenate([states, actions], axis=1)
    input_mean = inputs.mean(0).astype(np.float32)
    input_std = inputs.std(0).astype(np.float32)
    output_mean = deltas.mean(0).astype(np.float32)
    output_std = deltas.std(0).astype(np.float32)

    inp_t = torch.tensor((inputs - input_mean) / (input_std + 1e-8), dtype=torch.float32)
    out_t = torch.tensor((deltas - output_mean) / (output_std + 1e-8), dtype=torch.float32)

    dataset = TensorDataset(inp_t, out_t)
    val_size = max(1, int(args.val_fraction * len(dataset)))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = DynamicsNet(
        input_dim=5, output_dim=3, hidden_dim=args.hidden_dim, stochastic=args.stochastic
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, args.stochastic)
        val_loss = _val_epoch(model, val_loader, args.stochastic)
        print(f"Epoch {epoch:4d}/{args.epochs}  train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_mean": input_mean.tolist(),
                    "input_std": input_std.tolist(),
                    "output_mean": output_mean.tolist(),
                    "output_std": output_std.tolist(),
                    "stochastic": args.stochastic,
                    "hidden_dim": args.hidden_dim,
                    "dt": args.dt,
                },
                out_path,
            )
            print(f"  -> saved to {out_path}")

    print(f"Done. Best val loss: {best_val:.5f}")


if __name__ == "__main__":
    main()
