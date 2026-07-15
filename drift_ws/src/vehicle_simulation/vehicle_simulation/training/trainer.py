"""
Config-driven trainer for vehicle dynamics models (file-based run dirs).

Trains any model exposing the torch ``forward`` contract by rolling it over the
dataset windows and minimizing a single-step or multi-step trajectory loss (MSE or
Gaussian NLL). Device-agnostic and structured for later DistributedDataParallel
(the only DDP-specific seams are the sampler and an optional model wrap). Writes a
self-contained run directory: config, dataset spec, metrics, checkpoints, curves.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..data.dataset import Window, stack_windows
from ..data.spec import DatasetSpec
from .config import TrainConfig
from .run_dir import MetricsLogger, create_run_dir, write_meta, write_notes

_EPS = 1e-8


def pick_device(name: str) -> str:
    """Resolve ``"auto"`` to cuda when available, else echo the requested device."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def compute_normalization(windows: list[Window]) -> dict:
    """Input/output normalization + per-dim dynamic std from training windows."""
    batch = stack_windows(windows)
    dyn = batch["dyn_traj"]  # (N, H+1, 3)
    act = batch["actions"]  # (N, H, 2)
    dts = batch["dts"]  # (N, H)
    inputs = np.concatenate([dyn[:, :-1, :], act], axis=-1).reshape(-1, 6)
    deriv = ((dyn[:, 1:, :] - dyn[:, :-1, :]) / dts[..., None]).reshape(-1, 3)
    return {
        "input_mean": inputs.mean(0),
        "input_std": inputs.std(0) + 1e-6,
        "output_mean": deriv.mean(0),
        "output_std": deriv.std(0) + 1e-6,
        "dyn_std": dyn.reshape(-1, 3).std(0) + 1e-6,
    }


def rollout(model, dyn0: torch.Tensor, actions: torch.Tensor, dts: torch.Tensor):
    """
    Free-running rollout integrating the model's per-second derivative.

    Models with ``latent_dim > 0`` (internal actuator states) get their latents
    initialized quasi-steady from the first action and integrated alongside the
    dynamic state; the returned trajectory is stripped back to the dynamic dims.

    Returns the predicted dynamic trajectory ``(B, H+1, D)`` and the per-step
    *increment* covariances (``per-second cov · dt``; ``None`` for deterministic).
    """
    horizon = actions.shape[1]
    n_dyn = dyn0.shape[-1]
    d = dyn0
    if getattr(model, "latent_dim", 0):
        d = torch.cat([dyn0, model.init_latent(dyn0, actions[:, 0, :])], dim=-1)
    dyns = [d]
    covs = []
    for k in range(horizon):
        dt = dts[:, k].unsqueeze(-1)  # (B, 1)
        pred = model.derivative(d, actions[:, k, :])
        d = model.advance(d, actions[:, k, :], pred.mean, dt)
        covs.append(pred.cov * dt if pred.cov is not None else None)
        dyns.append(d)
    return torch.stack(dyns, dim=1)[..., :n_dyn], covs


def rollout_loss(dyn_pred, covs, dyn_gt, loss_type: str, dyn_std: torch.Tensor):
    """Per-sample loss (B,) over the rolled-out trajectory (excludes the given start)."""
    pred = dyn_pred[:, 1:, :]
    gt = dyn_gt[:, 1:, :]
    if loss_type == "mse":
        resid = (pred - gt) / dyn_std
        return (resid**2).mean(dim=(1, 2))
    if loss_type == "nll":
        if any(c is None for c in covs):
            raise ValueError("loss='nll' requires a stochastic model (cov is None)")
        cov = torch.stack(covs, dim=1).clamp_min(_EPS)  # (B,H,3)
        resid = pred - gt
        return (0.5 * (resid**2 / cov + torch.log(cov))).mean(dim=(1, 2))
    raise ValueError(f"Unknown loss: {loss_type!r}")


def _build_trainable_model(cfg: TrainConfig, normalization: dict, device: str):
    """Construct a fresh trainable model for ``cfg.model_type``."""
    m = cfg.model
    if cfg.model_type == "neural":
        from ..models.neural import NeuralDynamicsModel

        return NeuralDynamicsModel.new(
            normalization,
            hidden_dim=m.get("hidden_dim", 256),
            stochastic=m.get("stochastic", False),
            device=device,
        )
    if cfg.model_type == "dynamic":
        from ..models.dynamic import DynamicBicycleModel

        return DynamicBicycleModel.new(normalization, car=m.get("car", "drift_rc"), device=device)
    if cfg.model_type == "hybrid":
        from ..models.hybrid import PhysicsInformedModel

        return PhysicsInformedModel.new(
            normalization,
            car=m.get("car", "drift_rc"),
            n_terms=m.get("n_terms", 4),
            device=device,
        )
    raise ValueError(f"Trainer does not support model_type={cfg.model_type!r} yet")


def _to_tensors(windows: list[Window], device: str):
    """Stack windows and return (dyn0, actions, dts, dyn_traj) float32 tensors + regimes."""
    batch = stack_windows(windows)

    def t(key):
        return torch.as_tensor(batch[key], dtype=torch.float32)

    return (t("dyn0"), t("actions"), t("dts"), t("dyn_traj")), batch["regime"]


class Trainer:
    """Trains a dynamics model from windowed data into a self-contained run dir."""

    def __init__(
        self,
        cfg: TrainConfig,
        train_windows: list[Window],
        val_windows: list[Window],
        dataset_spec: DatasetSpec,
        run_base: str = "runs",
    ) -> None:
        self.cfg = cfg
        self.train_windows = train_windows
        self.val_windows = val_windows
        self.spec = dataset_spec
        self.run_base = run_base
        self.device = pick_device(cfg.device)

    def fit(self):
        """Run training and return the run directory path."""
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        norm = compute_normalization(self.train_windows)
        dyn_std = torch.as_tensor(norm["dyn_std"], dtype=torch.float32, device=self.device)
        model = _build_trainable_model(cfg, norm, self.device)
        opt = torch.optim.Adam(
            model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        (tr, regimes_tr) = _to_tensors(self.train_windows, self.device)
        (va, regimes_va) = _to_tensors(self.val_windows, self.device)
        loader = self._make_loader(tr, regimes_tr)

        run_dir = create_run_dir(self.run_base, cfg.model_type, cfg.tag)
        cfg.to_yaml(str(run_dir / "config.yaml"))
        self.spec.to_yaml(str(run_dir / "dataset_spec.yaml"))
        write_notes(run_dir, cfg.notes or self.spec.notes)
        write_meta(run_dir, device=self.device, seed=cfg.seed)
        logger = MetricsLogger(run_dir)

        best = float("inf")
        patience = 0
        for epoch in range(1, cfg.epochs + 1):
            model.set_train_mode(True)
            train_losses = []
            for dyn0, actions, dts, dyn_traj in loader:
                dyn0 = dyn0.to(self.device)
                actions = actions.to(self.device)
                dts = dts.to(self.device)
                dyn_traj = dyn_traj.to(self.device)
                dyn_pred, covs = rollout(model, dyn0, actions, dts)
                loss = rollout_loss(dyn_pred, covs, dyn_traj, cfg.loss, dyn_std).mean()
                opt.zero_grad()
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
                opt.step()
                train_losses.append(float(loss.item()))

            row = {"epoch": epoch, "train_loss": float(np.mean(train_losses))}
            if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
                model.set_train_mode(False)
                val_loss, per_regime = self._validate(model, va, regimes_va, dyn_std)
                row["val_loss"] = val_loss
                for regime, value in per_regime.items():
                    row[f"val_{regime}"] = value
                torch.save(model.to_checkpoint(), run_dir / "model_last.pt")
                if val_loss < best - cfg.early_stop_min_delta:
                    best = val_loss
                    patience = 0
                    torch.save(model.to_checkpoint(), run_dir / "model.pt")
                else:
                    patience += 1
            logger.log(row)
            if patience >= cfg.early_stop_patience:
                break

        if not (run_dir / "model.pt").exists():
            torch.save(model.to_checkpoint(), run_dir / "model.pt")
        logger.plot(run_dir / "loss_curve.png")
        return run_dir

    def _make_loader(self, tensors, regimes) -> DataLoader:
        # DDP seam: when cfg.ddp is wired up (launched via torchrun), swap the
        # sampler below for a DistributedSampler and wrap the model's net in
        # DistributedDataParallel. Single-GPU/CPU path is the default.
        dataset = TensorDataset(*tensors)
        kw = {
            "batch_size": self.cfg.batch_size,
            "num_workers": self.cfg.num_workers,
            "pin_memory": self.device == "cuda",
        }
        if self.cfg.regime_weights:
            weights = np.array(
                [self.cfg.regime_weights.get(str(r), 1.0) for r in regimes], dtype=np.float64
            )
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights), num_samples=len(dataset), replacement=True
            )
            return DataLoader(dataset, sampler=sampler, **kw)
        return DataLoader(dataset, shuffle=True, **kw)

    def _validate(self, model, tensors, regimes, dyn_std):
        """Overall + per-regime validation loss (per-sample, batched, no grad)."""
        dyn0, actions, dts, dyn_traj = tensors
        per_sample = []
        with torch.no_grad():
            for i in range(0, len(dyn0), self.cfg.batch_size):
                sl = slice(i, i + self.cfg.batch_size)
                dyn_pred, covs = rollout(
                    model,
                    dyn0[sl].to(self.device),
                    actions[sl].to(self.device),
                    dts[sl].to(self.device),
                )
                losses = rollout_loss(
                    dyn_pred, covs, dyn_traj[sl].to(self.device), self.cfg.loss, dyn_std
                )
                per_sample.append(losses.cpu().numpy())
        per_sample = np.concatenate(per_sample) if per_sample else np.array([])
        overall = float(per_sample.mean()) if len(per_sample) else float("nan")
        per_regime = {}
        for regime in sorted(set(map(str, regimes))):
            mask = np.array([str(r) == regime for r in regimes])
            if mask.any():
                per_regime[regime] = float(per_sample[mask].mean())
        return overall, per_regime
