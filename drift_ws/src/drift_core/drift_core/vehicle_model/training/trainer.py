"""
Config-driven trainer for vehicle dynamics models (file-based run dirs).

Trains any model exposing the torch ``forward`` contract by rolling it over the
dataset windows and minimizing a single-step or multi-step trajectory loss (MSE or
Gaussian NLL). Device-agnostic and structured for later DistributedDataParallel
(the only DDP-specific seams are the sampler and an optional model wrap). Writes a
self-contained run directory: config, dataset spec, metrics, checkpoints, curves.
"""

import inspect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ...paths import vehicle_model_runs_dir
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


def rollout(model, dyn0: torch.Tensor, actions: torch.Tensor, dts: torch.Tensor, sample=False):
    """
    Free-running rollout integrating the model's per-second derivative.

    Models with ``latent_dim > 0`` (internal actuator states) get their latents
    initialized quasi-steady from the first action and integrated alongside the
    dynamic state; the returned trajectory is stripped back to the dynamic dims.

    With ``sample=True`` a stochastic model's rollout is Euler–Maruyama: each step
    adds ``sqrt(cov·dt)·eps`` to the dims the covariance covers (latents get no
    noise), reparameterized so gradients flow through both mean and variance —
    the next step's derivative then sees the *sampled* state, which is how
    uncertainty compounds through the dynamics.

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
        if sample and pred.cov is not None:
            inc = (pred.cov * dt).clamp_min(0.0)
            noise = inc.sqrt() * torch.randn_like(pred.cov)
            d = torch.cat([d[..., : noise.shape[-1]] + noise, d[..., noise.shape[-1] :]], dim=-1)
        covs.append(pred.cov * dt if pred.cov is not None else None)
        dyns.append(d)
    return torch.stack(dyns, dim=1)[..., :n_dyn], covs


def nll_particle_loss(
    model, dyn0, actions, dts, dyn_traj, dyn_std, particles: int, meas_std=(0.03, 0.03, 0.1)
):
    """
    Particle NLL by *ensemble moment matching* (per-window losses, shape ``(B,)``).

    Samples ``particles`` Euler–Maruyama rollouts per window, forms the ensemble's
    per-step mean and variance, and evaluates the recorded trajectory under that
    Gaussian. This calibrates the *fan* directly: a fan wider than the data's actual
    deviation is punished by ``log V``, a narrower one by the Mahalanobis term, and
    the equilibrium is a spread that statistically matches the recorded errors.

    Two rejected alternatives, for the record: the NLL around the deterministic mean
    trajectory over-inflates the diffusion (it absorbs accumulated rollout error and
    measurement noise as per-step noise, which double-counts once sampling
    re-compounds it — the fans came out 3-10x too wide); and the paper's literal
    per-particle form (data-vs-sampled-path residual scaled by the one-step increment
    variance) is an inconsistent likelihood with a runaway feedback on this dataset —
    a wider fan makes its own residuals larger, which the fit "explains" with still
    more variance. Gradients flow into the diffusion through the reparameterized
    samples in both moments.
    """
    k = int(particles)
    if k < 2:
        raise ValueError("nll_particle_loss needs particles >= 2 to estimate a variance")
    dyn_pred, _ = rollout(
        model,
        dyn0.repeat_interleave(k, dim=0),
        actions.repeat_interleave(k, dim=0),
        dts.repeat_interleave(k, dim=0),
        sample=True,
    )
    preds = dyn_pred.view(-1, k, *dyn_pred.shape[1:])  # (B, K, H+1, 3)
    mu = preds.mean(dim=1)[:, 1:, :]
    # Observation model: recorded state = true state + tracker noise. Adding the
    # measurement variance here keeps the fit from inflating the *process* diffusion
    # to cover sensor jitter (which never shrinks, not even for a parked car).
    r_meas = torch.as_tensor(meas_std, dtype=preds.dtype, device=preds.device) ** 2
    var = preds.var(dim=1, unbiased=True)[:, 1:, :] + r_meas
    resid = dyn_traj[:, 1:, :] - mu
    return (0.5 * (resid**2 / var + torch.log(var))).mean(dim=(1, 2))


def rollout_loss(
    dyn_pred, covs, dyn_gt, loss_type: str, dyn_std: torch.Tensor, meas_std=(0.03, 0.03, 0.1)
):
    """Per-sample loss (B,) over the rolled-out trajectory (excludes the given start)."""
    pred = dyn_pred[:, 1:, :]
    gt = dyn_gt[:, 1:, :]
    if loss_type == "mse":
        resid = (pred - gt) / dyn_std
        return (resid**2).mean(dim=(1, 2))
    if loss_type == "nll":
        if any(c is None for c in covs):
            raise ValueError("loss='nll' requires a stochastic model (cov is None)")
        # Observation model, same as the particle loss: the residual contains tracker
        # noise on top of the process increment. Without this floor, states where the
        # model's cov is legitimately ~zero (the diffusion is slip-gated; a rolling
        # car has almost no process noise) divide the measurement noise by ~0 and
        # dominate the loss — the physics then bends to appease them.
        r_meas = torch.as_tensor(meas_std, dtype=pred.dtype, device=pred.device) ** 2
        cov = torch.stack(covs, dim=1) + r_meas  # (B,H,3)
        resid = pred - gt
        return (0.5 * (resid**2 / cov + torch.log(cov))).mean(dim=(1, 2))
    raise ValueError(f"Unknown loss: {loss_type!r}")


def nll_paper_particle_loss(
    model, dyn0, actions, dts, dyn_traj, dyn_std, particles: int, meas_std=(0.03, 0.03, 0.1)
):
    """
    The paper's literal particle NLL (Djeumou et al. Eq. 3), per-window losses ``(B,)``.

    Draws ``particles`` Euler–Maruyama rollouts per window and scores the *recorded*
    trajectory against **each sampled path individually** — residual ``x̄_k − x̃_k^(p)``
    weighted by that path's own per-step diffusion ``Σ_θ(x̃_k^(p), u_k)·dt`` — then
    averages the per-particle NLLs, a Monte-Carlo estimate of the expectation over
    sampled trajectories in Eq. 3. (Reuses the ``rollout_loss`` NLL kernel on each
    sampled path, since that is exactly the per-path weighted-norm + log-det term.)

    Contrast :func:`nll_particle_loss`, which instead matches the ensemble's *moments*
    (mean + empirical variance). The per-particle form is faithful to the paper but
    self-referential: the diffusion both scales the residual *and* generates the sampled
    path the residual is measured against, so with ``train_scope="all"`` on a small
    dataset it can inflate the fan without bound (a wider fan lowers its own log-det
    penalty faster than it grows the residuals). Pair it with ``train_scope="diffusion"``
    — physics frozen — which is how the paper trains the SDE as a single calibration.

    Deviations from Eq. 3, both deliberate: the ``0.5`` / per-step / per-dim
    normalization is the repo's convention (a global positive constant vs. Eq. 3's raw
    sum over k — identical optimum, comparable across the three NLL paths); and ``meas_std``
    adds a variance floor, as in the other two losses, so the slip-gated Σ (which can
    reach ~0) keeps the log-likelihood finite. Unlike moment matching, this works with a
    single particle (no empirical variance is needed), so ``particles >= 1``.
    """
    k = int(particles)
    if k < 1:
        raise ValueError("nll_paper_particle_loss needs particles >= 1")
    dyn_pred, covs = rollout(
        model,
        dyn0.repeat_interleave(k, dim=0),
        actions.repeat_interleave(k, dim=0),
        dts.repeat_interleave(k, dim=0),
        sample=True,
    )
    gt = dyn_traj.repeat_interleave(k, dim=0)
    per_particle = rollout_loss(dyn_pred, covs, gt, "nll", dyn_std, meas_std=meas_std)  # (B*k,)
    return per_particle.view(-1, k).mean(dim=1)


def _build_trainable_model(cfg: TrainConfig, normalization: dict, device: str):
    """
    Construct a fresh trainable model for ``cfg.model_type``.

    ``cfg.model`` is forwarded to the model's ``new()`` wholesale. It used to be
    enumerated key by key here, which silently dropped every other key: a config asking
    for ``bounded_params: false`` trained a *bounded* model for 120 epochs and only the
    checkpoint's own record of the flag revealed it. Unknown keys are rejected below
    rather than swallowed, so a typo fails at construction instead of at analysis time.
    """
    m = dict(cfg.model)
    if cfg.model_type == "neural":
        from ..models.neural import NeuralDynamicsModel

        model_cls = NeuralDynamicsModel
    elif cfg.model_type == "dynamic":
        from ..models.dynamic import DynamicBicycleModel

        model_cls = DynamicBicycleModel
    elif cfg.model_type == "hybrid":
        from ..models.hybrid import PhysicsInformedModel

        model_cls = PhysicsInformedModel
    else:
        raise ValueError(f"Trainer does not support model_type={cfg.model_type!r} yet")

    accepted = set(inspect.signature(model_cls.new).parameters)
    unknown = sorted(set(m) - accepted)
    if unknown:
        raise ValueError(
            f"{cfg.model_type} model config has unknown key(s) {unknown}; "
            f"{model_cls.__name__}.new accepts {sorted(accepted - {'cls', 'normalization'})}"
        )
    return model_cls.new(normalization, device=device, **m)


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
        run_base: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.train_windows = train_windows
        self.val_windows = val_windows
        self.spec = dataset_spec
        self.run_base = run_base or vehicle_model_runs_dir()
        self.device = pick_device(cfg.device)

    def fit(self):
        """Run training and return the run directory path."""
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        norm = compute_normalization(self.train_windows)
        dyn_std = torch.as_tensor(norm["dyn_std"], dtype=torch.float32, device=self.device)
        if cfg.init_from:
            from ..models.loader import load_model_from_run

            model = load_model_from_run(cfg.init_from, device=self.device)
        else:
            model = _build_trainable_model(cfg, norm, self.device)
        if cfg.nll_estimator not in ("moment", "particle"):
            raise ValueError(f"Unknown nll_estimator: {cfg.nll_estimator!r}")
        if cfg.train_scope == "diffusion":
            # Freeze the physics; calibrate only the diffusion head. This is what makes
            # the particle NLL safe on a small dataset: sampling cannot push the tire
            # fit toward the degenerate over-grippy solution when the tires cannot move.
            params = model.diffusion_parameters()
        elif cfg.train_scope == "all":
            params = model.trainable_parameters()
        else:
            raise ValueError(f"Unknown train_scope: {cfg.train_scope!r}")
        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

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
                if cfg.loss == "nll" and cfg.nll_particles > 0:
                    loss = self._particle_loss(model, dyn0, actions, dts, dyn_traj, dyn_std).mean()
                else:
                    dyn_pred, covs = rollout(model, dyn0, actions, dts)
                    loss = rollout_loss(
                        dyn_pred, covs, dyn_traj, cfg.loss, dyn_std, meas_std=cfg.nll_meas_std
                    ).mean()
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

    def _particle_loss(self, model, dyn0, actions, dts, dyn_traj, dyn_std):
        """Per-window NLL over sampled rollouts, dispatched on ``cfg.nll_estimator``."""
        cfg = self.cfg
        fn = nll_paper_particle_loss if cfg.nll_estimator == "particle" else nll_particle_loss
        return fn(
            model,
            dyn0,
            actions,
            dts,
            dyn_traj,
            dyn_std,
            cfg.nll_particles,
            meas_std=cfg.nll_meas_std,
        )

    def _validate(self, model, tensors, regimes, dyn_std):
        """Overall + per-regime validation loss (per-sample, batched, no grad)."""
        dyn0, actions, dts, dyn_traj = tensors
        per_sample = []
        with torch.no_grad():
            for i in range(0, len(dyn0), self.cfg.batch_size):
                sl = slice(i, i + self.cfg.batch_size)
                batch = (
                    dyn0[sl].to(self.device),
                    actions[sl].to(self.device),
                    dts[sl].to(self.device),
                    dyn_traj[sl].to(self.device),
                )
                if self.cfg.loss == "nll" and self.cfg.nll_particles > 0:
                    losses = self._particle_loss(model, *batch, dyn_std)
                else:
                    dyn_pred, covs = rollout(model, *batch[:3])
                    losses = rollout_loss(
                        dyn_pred,
                        covs,
                        batch[3],
                        self.cfg.loss,
                        dyn_std,
                        meas_std=self.cfg.nll_meas_std,
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
