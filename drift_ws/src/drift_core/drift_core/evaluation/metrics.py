"""
Quantitative model evaluation (eval methods 1 & 2).

Rolls a model over evaluation windows and reports overall error, a per-regime
breakdown (where the model works / fails), and a per-horizon error-growth curve
(the most informative signal for multi-step / sim-to-real quality).
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import Window, stack_windows
from ..models.integrate import integrate_pose
from ..training.trainer import rollout


@dataclass
class EvalReport:
    """Overall + per-regime + per-horizon evaluation metrics."""

    overall: dict
    per_regime: dict = field(default_factory=dict)
    per_horizon: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        """Write the report to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _pose_rollout(pose0: np.ndarray, dyn_pred: np.ndarray, dts: np.ndarray) -> np.ndarray:
    """Integrate predicted body velocities into a pose trajectory (N, H+1, 3)."""
    pose = pose0.copy()
    poses = [pose]
    horizon = dyn_pred.shape[1] - 1
    for k in range(horizon):
        # Semi-implicit Euler, matching TrainableModel._integrate: advance the pose
        # over step k with the *updated* (end-of-step) velocities dyn_pred[:, k + 1].
        pose = integrate_pose(
            pose, dyn_pred[:, k + 1, 0], dyn_pred[:, k + 1, 1], dyn_pred[:, k + 1, 2], dts[:, k]
        )
        poses.append(pose)
    return np.stack(poses, axis=1)


def _metrics_for(dyn_pred, dyn_gt, pose_pred, pose_gt, covs_np=None) -> dict:
    err = (dyn_pred[:, 1:, :] - dyn_gt[:, 1:, :]) ** 2  # (N, H, 3)
    pos_err = np.sum((pose_pred[:, 1:, :2] - pose_gt[:, 1:, :2]) ** 2, axis=-1)  # (N, H)
    out = {
        "n": int(dyn_pred.shape[0]),
        "single_step_mse": float(err[:, 0, :].mean()),
        "rollout_mse": float(err.mean()),
        "rollout_pose_rmse": float(np.sqrt(pos_err.mean())),
    }
    if covs_np is not None:
        cov = np.clip(covs_np, 1e-8, None)
        out["rollout_nll"] = float((0.5 * (err / cov + np.log(cov))).mean())
    return out


def evaluate_model(model, windows: list[Window], *, device: str = "cpu") -> EvalReport:
    """
    Evaluate ``model`` on ``windows``: overall, per-regime, and per-horizon errors.

    Errors are physical MSE on the canonical dynamic state plus pose RMSE from
    integrating the predicted velocities. Stochastic models also get an NLL.
    """
    batch = stack_windows(windows)
    regimes = batch["regime"]
    dyn_gt = batch["dyn_traj"]
    pose_gt = batch["pose_traj"]
    dts = batch["dts"]

    model.set_train_mode(False)
    with torch.no_grad():
        dyn_pred_t, covs = rollout(
            model,
            torch.as_tensor(batch["dyn0"], dtype=torch.float32, device=device),
            torch.as_tensor(batch["actions"], dtype=torch.float32, device=device),
            torch.as_tensor(dts, dtype=torch.float32, device=device),
        )
    dyn_pred = dyn_pred_t.cpu().numpy()
    covs_np = None
    if all(c is not None for c in covs):
        covs_np = torch.stack(covs, dim=1).cpu().numpy()  # (N, H, 3)

    pose_pred = _pose_rollout(pose_gt[:, 0, :], dyn_pred, dts)

    overall = _metrics_for(dyn_pred, dyn_gt, pose_pred, pose_gt, covs_np)

    err = (dyn_pred[:, 1:, :] - dyn_gt[:, 1:, :]) ** 2
    per_horizon = {int(k + 1): float(err[:, k, :].mean()) for k in range(err.shape[1])}

    per_regime = {}
    for regime in sorted(set(map(str, regimes))):
        mask = np.array([str(r) == regime for r in regimes])
        if mask.any():
            per_regime[regime] = _metrics_for(
                dyn_pred[mask],
                dyn_gt[mask],
                pose_pred[mask],
                pose_gt[mask],
                covs_np[mask] if covs_np is not None else None,
            )
    return EvalReport(overall=overall, per_regime=per_regime, per_horizon=per_horizon)
