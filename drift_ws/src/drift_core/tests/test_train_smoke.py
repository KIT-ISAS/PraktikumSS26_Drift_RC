"""Phase 2: end-to-end training smoke test (torch-guarded)."""

import csv

import pytest

pytest.importorskip("torch")

from drift_core.data.spec import DatasetSpec  # noqa: E402
from drift_core.training.config import TrainConfig  # noqa: E402
from drift_core.training.trainer import Trainer  # noqa: E402


def _cfg(**kw) -> TrainConfig:
    base = dict(
        model_type="neural",
        model={"hidden_dim": 32},
        epochs=6,
        batch_size=64,
        lr=1e-2,
        val_every=1,
        device="cpu",
        tag="smoke",
    )
    base.update(kw)
    return TrainConfig(**base)


def test_training_runs_and_writes_artifacts(linear_windows, tmp_path):
    train, val = linear_windows
    spec = DatasetSpec(bags=["synthetic"], horizon=5, notes="smoke test")
    run_dir = Trainer(_cfg(), train, val, spec, run_base=str(tmp_path)).fit()

    for name in ("config.yaml", "dataset_spec.yaml", "notes.md", "metrics.csv", "model.pt"):
        assert (run_dir / name).exists(), f"missing {name}"

    rows = list(csv.DictReader(open(run_dir / "metrics.csv")))
    assert rows
    first = float(rows[0]["train_loss"])
    last = float(rows[-1]["train_loss"])
    assert last <= first  # loss should not increase on learnable dynamics


def test_multi_step_horizon_trains(linear_windows, tmp_path):
    # horizon > 1 rollout loss path (windows already have horizon 5).
    train, val = linear_windows
    spec = DatasetSpec(bags=["synthetic"], horizon=5)
    run_dir = Trainer(_cfg(epochs=3), train, val, spec, run_base=str(tmp_path)).fit()
    assert (run_dir / "model.pt").exists()


def test_stochastic_nll_path_runs(linear_windows, tmp_path):
    train, val = linear_windows
    spec = DatasetSpec(bags=["synthetic"], horizon=5)
    cfg = _cfg(model={"hidden_dim": 32, "stochastic": True}, loss="nll", epochs=3)
    run_dir = Trainer(cfg, train, val, spec, run_base=str(tmp_path)).fit()
    assert (run_dir / "model.pt").exists()
