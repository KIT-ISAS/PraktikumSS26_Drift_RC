"""
CLI to evaluate a trained model: metrics + per-regime + per-horizon + MCAP.

Usage::

    python -m vehicle_simulation.evaluation.evaluate \
        --run-dir runs/<...> --bags drift_ws/rosbag2_2026_06_03-11_43_14 \
        --mcap-bag drift_ws/rosbag2_2026_06_03-11_43_14

Writes ``<run_dir>/eval/metrics.json`` (+ plots if matplotlib is installed) and,
when ``--mcap-bag`` is given, a recorded-vs-predicted ``.mcap`` for Foxglove.
"""

import argparse
from pathlib import Path

from ..data.spec import DatasetSpec, build_dataset_from_spec, default_filters
from ..models.loader import load_model_from_run
from .metrics import evaluate_model
from .plots import save_all


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained vehicle model.")
    p.add_argument("--run-dir", required=True, help="trained run directory")
    p.add_argument("--dataset-spec", help="DatasetSpec YAML for eval data")
    p.add_argument("--bags", nargs="+", help="bags to build an eval dataset (if no spec)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--mcap-bag", help="bag to render a recorded-vs-predicted MCAP")
    p.add_argument("--mcap-start", type=int, default=0)
    p.add_argument("--mcap-horizon", type=int, default=200)
    return p.parse_args(argv)


def _eval_spec(args) -> DatasetSpec:
    if args.dataset_spec:
        return DatasetSpec.from_yaml(args.dataset_spec)
    return DatasetSpec(
        bags=list(args.bags),
        regimes_included=["straight", "corner", "drift"],
        filters=default_filters(),
    )


def main(argv=None) -> None:
    """Run evaluation methods 1-3 and write artifacts into the run dir's eval/."""
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_from_run(str(run_dir), device=args.device)

    if not (args.dataset_spec or args.bags):
        raise SystemExit("Provide --dataset-spec or --bags for evaluation data")
    train, val, test, _ = build_dataset_from_spec(_eval_spec(args))
    eval_windows = test or val or train
    report = evaluate_model(model, eval_windows, device=args.device)
    report.save(str(eval_dir / "metrics.json"))
    plots = save_all(report, str(eval_dir))

    print("overall:", report.overall)
    for regime, m in report.per_regime.items():
        print(
            f"  {regime}: rollout_mse={m['rollout_mse']:.4f} "
            f"pose_rmse={m['rollout_pose_rmse']:.4f}"
        )
    print(f"wrote {eval_dir / 'metrics.json'}" + (f" + {len(plots)} plots" if plots else ""))

    if args.mcap_bag:
        from ..data.rosbag_reader import read_driving_data
        from .mcap_writer import rollout_from_recording, write_eval_mcap

        df = read_driving_data(args.mcap_bag)
        times, rec, pred, actions = rollout_from_recording(
            model, df, start_idx=args.mcap_start, horizon=args.mcap_horizon
        )
        out = write_eval_mcap(
            str(eval_dir / "recorded_vs_predicted.mcap"), times, rec, pred, actions=actions
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
