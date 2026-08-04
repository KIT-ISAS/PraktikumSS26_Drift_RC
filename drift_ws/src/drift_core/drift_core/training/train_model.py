"""
CLI to train a vehicle dynamics model into a self-contained run directory.

Usage::

    # From a dataset-spec YAML + a train-config YAML:
    python -m drift_core.training.train_model \
        --dataset-spec spec.yaml --config train.yaml --tag v1

    # Quick start straight from bags (default spec + config):
    python -m drift_core.training.train_model \
        --bags drift_ws/rosbag2_2026_06_03-11_43_14 --tag quickstart \
        --notes "first neural model, drift+corner only"

The dataset recipe (DatasetSpec) and hyperparameters (TrainConfig) are saved into
the run dir so the run is fully reproducible.
"""

import argparse

from ..data.spec import DatasetSpec, build_dataset_from_spec, default_filters
from .config import TrainConfig
from .trainer import Trainer


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a vehicle dynamics model.")
    p.add_argument("--config", help="TrainConfig YAML (defaults used if omitted)")
    p.add_argument("--dataset-spec", help="DatasetSpec YAML")
    p.add_argument(
        "--bags", nargs="+", help="rosbag2 dirs (build a default spec if no --dataset-spec)"
    )
    p.add_argument(
        "--runs-dir", help="base directory for run dirs (default: drift_ws/runs/vehicle_model)"
    )
    p.add_argument("--tag", help="short run label (overrides config)")
    p.add_argument("--notes", help="free-text notes (overrides config)")
    p.add_argument("--epochs", type=int, help="override config.epochs")
    p.add_argument("--device", help="override config.device (auto|cpu|cuda)")
    return p.parse_args(argv)


def _load_spec(args: argparse.Namespace) -> DatasetSpec:
    if args.dataset_spec:
        return DatasetSpec.from_yaml(args.dataset_spec)
    if args.bags:
        return DatasetSpec(
            bags=list(args.bags),
            regimes_included=["straight", "corner", "drift"],
            filters=default_filters(),
        )
    raise SystemExit("Provide --dataset-spec or --bags")


def main(argv=None) -> None:
    """Entry point: build the dataset, train, and print the run directory."""
    args = _parse_args(argv)
    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    if args.tag is not None:
        cfg.tag = args.tag
    if args.notes is not None:
        cfg.notes = args.notes
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.device is not None:
        cfg.device = args.device

    spec = _load_spec(args)
    train, val, test, spec_out = build_dataset_from_spec(spec)
    print(f"dataset: {spec_out.n_windows} windows {spec_out.n_windows_per_regime}")
    print(f"split: train={len(train)} val={len(val)} test={len(test)}")
    if not train or not val:
        raise SystemExit("Empty train/val split — adjust the dataset spec.")

    run_dir = Trainer(cfg, train, val, spec_out, run_base=args.runs_dir).fit()
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
