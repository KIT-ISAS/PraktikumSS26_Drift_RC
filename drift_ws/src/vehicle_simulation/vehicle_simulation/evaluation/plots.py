"""Matplotlib plots for evaluation (optional; needs the ``eval`` extra)."""

from pathlib import Path


def _pyplot():
    """Return matplotlib.pyplot (Agg backend) or None if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def plot_per_horizon(report, out_path: str) -> bool:
    """Plot the rollout error-growth curve (MSE vs horizon step)."""
    plt = _pyplot()
    if plt is None or not report.per_horizon:
        return False
    steps = sorted(report.per_horizon)
    fig, ax = plt.subplots()
    ax.plot(steps, [report.per_horizon[s] for s in steps], marker="o")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("dynamic MSE")
    ax.set_title("Error growth over horizon")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_regime_bars(report, out_path: str, metric: str = "rollout_mse") -> bool:
    """Bar chart of a metric per regime (where the model works / fails)."""
    plt = _pyplot()
    if plt is None or not report.per_regime:
        return False
    regimes = sorted(report.per_regime)
    values = [report.per_regime[r].get(metric, float("nan")) for r in regimes]
    fig, ax = plt.subplots()
    ax.bar(regimes, values)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by regime")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def save_all(report, out_dir: str) -> list[str]:
    """Write all evaluation plots into ``out_dir``; returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    if plot_per_horizon(report, str(out / "per_horizon.png")):
        written.append(str(out / "per_horizon.png"))
    if plot_regime_bars(report, str(out / "regime_mse.png")):
        written.append(str(out / "regime_mse.png"))
    return written
