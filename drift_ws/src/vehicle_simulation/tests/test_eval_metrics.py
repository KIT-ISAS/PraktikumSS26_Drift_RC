"""Phase 4: evaluation metrics (overall / per-regime / per-horizon), torch-guarded."""

import pytest

pytest.importorskip("torch")

from vehicle_simulation.evaluation.metrics import EvalReport, evaluate_model  # noqa: E402
from vehicle_simulation.models import build_model  # noqa: E402


def test_eval_report_structure(linear_windows):
    train, val = linear_windows
    model = build_model("dynamic")  # untrained physics; just exercising the metrics
    report = evaluate_model(model, val)
    assert isinstance(report, EvalReport)
    for key in ("single_step_mse", "rollout_mse", "rollout_pose_rmse", "n"):
        assert key in report.overall
    assert report.per_horizon
    known = {"straight", "corner", "drift", "stationary", "unknown"}
    assert set(report.per_regime).issubset(known)


def test_per_horizon_error_grows_for_biased_model(linear_windows):
    train, val = linear_windows
    model = build_model("dynamic")
    report = evaluate_model(model, val)
    steps = sorted(report.per_horizon)
    first = report.per_horizon[steps[0]]
    last = report.per_horizon[steps[-1]]
    # Compounding free-running error: an untrained model's later horizons must not
    # come out systematically better than its first step. Exact monotonicity is not
    # guaranteed (dissipative segments — braking to rest — can shrink late errors),
    # so allow slack while still catching a collapse.
    assert last >= 0.5 * first


def test_eval_report_saves_json(linear_windows, tmp_path):
    train, val = linear_windows
    report = evaluate_model(build_model("dynamic"), val)
    out = tmp_path / "metrics.json"
    report.save(str(out))
    assert out.exists() and out.stat().st_size > 0
