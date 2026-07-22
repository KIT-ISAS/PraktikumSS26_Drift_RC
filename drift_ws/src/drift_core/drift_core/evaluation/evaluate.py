"""
CLI to evaluate a trained model: metrics + per-regime + per-horizon + MCAP.

Usage::

    python -m drift_core.evaluation.evaluate \
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
    p.add_argument(
        "--mcap-start",
        type=int,
        nargs="+",
        default=[0],
        help="start indices into the recording; each writes its own MCAP",
    )
    p.add_argument("--mcap-horizon", type=int, default=200)
    p.add_argument(
        "--mcap-samples",
        type=int,
        default=0,
        help="additionally draw N stochastic SDE paths per MCAP (needs a stochastic model)",
    )
    p.add_argument(
        "--mcap-seed",
        type=int,
        default=None,
        help="seed the sampled-fan RNG so a fan is reproducible across runs",
    )
    p.add_argument(
        "--mcap-no-sanitize",
        action="store_true",
        help="roll the MCAP over raw samples, without screening for localization jumps",
    )
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
    # Sampled-fan MCAPs need the model's sampling armed; metrics always use the mean.
    model = load_model_from_run(str(run_dir), device=args.device, stochastic=args.mcap_samples > 0)
    if args.mcap_samples > 0 and not getattr(model, "_can_sample", False):
        # A deterministic checkpoint would otherwise yield K identical copies of the
        # mean and an all-zero spread — the exact failure this tool exists to catch,
        # dressed up as a clean result. Refuse it instead.
        raise SystemExit("--mcap-samples needs a checkpoint trained with a diffusion head")

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
        from .mcap_writer import rollout_from_recording, sampled_rollout_fan, write_eval_mcap

        df = read_driving_data(args.mcap_bag)
        written = 0
        for start in args.mcap_start:
            # One glitchy start (a localization jump at the seed / next sample) raises
            # from the rollout; with a batch of starts, skip it and keep going rather
            # than aborting the whole invocation and silently leaving the rest unwritten.
            try:
                sampled = None
                if args.mcap_samples > 0:
                    times, rec, pred, actions, sampled = sampled_rollout_fan(
                        model,
                        df,
                        start_idx=start,
                        horizon=args.mcap_horizon,
                        samples=args.mcap_samples,
                        sanitize=not args.mcap_no_sanitize,
                        seed=args.mcap_seed,
                    )
                else:
                    times, rec, pred, actions = rollout_from_recording(
                        model,
                        df,
                        start_idx=start,
                        horizon=args.mcap_horizon,
                        sanitize=not args.mcap_no_sanitize,
                    )
            except ValueError as e:
                print(f"skipped start {start}: {e}")
                continue
            name = (
                "recorded_vs_predicted.mcap"
                if len(args.mcap_start) == 1
                else f"recorded_vs_predicted_t{start}.mcap"
            )
            out = write_eval_mcap(
                str(eval_dir / name), times, rec, pred, actions=actions, samples=sampled
            )
            written += 1
            print(f"wrote {out}")
        print(f"wrote {written}/{len(args.mcap_start)} MCAP(s)")


if __name__ == "__main__":
    main()
