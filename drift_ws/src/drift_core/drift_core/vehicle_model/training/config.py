"""Training configuration (dataclass + YAML), saved verbatim into each run dir."""

from dataclasses import asdict, dataclass, field

import yaml


@dataclass
class TrainConfig:
    """
    All hyperparameters for one training run.

    The dataset recipe lives separately in a
    :class:`~drift_core.vehicle_model.data.spec.DatasetSpec`
    (saved as ``dataset_spec.yaml``); this captures everything else. The rollout
    horizon is taken from the dataset windows, so single-step vs multi-step is a
    property of the dataset, not this config.
    """

    model_type: str = "neural"  # registry name: neural | dynamic | hybrid
    model: dict = field(default_factory=dict)  # constructor kwargs (hidden_dim, stochastic, ...)
    loss: str = "mse"  # "mse" (deterministic) | "nll" (stochastic)
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    regime_weights: dict | None = None  # {regime: weight} for weighted sampling
    val_every: int = 5
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4
    device: str = "auto"  # "auto" -> cuda if available, else cpu
    seed: int = 0
    num_workers: int = 0
    tag: str = ""  # short label for the run-dir name
    notes: str = ""  # free-text human notes copied into the run dir

    def to_dict(self) -> dict:
        """Plain-dict view for YAML."""
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "TrainConfig":
        """Build a config from a parsed mapping (ignores unknown keys)."""
        fields = {f for f in TrainConfig.__dataclass_fields__}
        return TrainConfig(**{k: v for k, v in data.items() if k in fields})

    def to_yaml(self, path: str) -> None:
        """Write the config to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @staticmethod
    def from_yaml(path: str) -> "TrainConfig":
        """Read a config from a YAML file."""
        with open(path) as f:
            return TrainConfig.from_dict(yaml.safe_load(f))
