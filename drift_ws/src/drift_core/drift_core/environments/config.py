"""
Curriculum stage configuration (dataclasses + YAML), one file per training stage.

A :class:`StageConfig` bundles everything one curriculum stage needs: the
environment (track, model, spawn, reward, termination) and the SAC training
hyperparameters. The three packaged stages live in ``environments/configs/`` and
load by name (``StageConfig.load("stage1_path_follow")``), mirroring how
``load_track`` resolves packaged track names.
"""

from dataclasses import asdict, dataclass, field, fields
import importlib.resources
from pathlib import Path

import yaml

from .rewards import RewardConfig, TerminationConfig


def _from_mapping(cls, data: dict):
    """Instantiate a flat dataclass from a mapping, rejecting unknown keys."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


@dataclass
class SpawnConfig:
    """Episode spawn distribution along the centerline."""

    speed: float = 1.0  # m/s initial forward speed
    speed_noise_std: float = 0.1  # m/s
    standstill_prob: float = 0.0  # probability of spawning parked (vx = 0) instead
    lateral_noise_std: float = 0.05  # m, clipped to ±2 std
    heading_noise_std: float = 0.15  # rad


@dataclass
class EnvConfig:
    """Everything :class:`DriftEnv` needs for one stage."""

    track: str = "figure_8"  # packaged track name or a YAML path (load_track semantics)
    model_type: str = "arcade"  # forwarded to build_model (also accepts a run dir)
    model_config: str | None = None  # optional model YAML (build_model's config)
    model_kwargs: dict = field(default_factory=dict)
    dt: float = 0.02  # s — 50 Hz, matching the real control loop
    points_per_segment: int = 20  # centerline discretization (BezierTrack.discretize)
    max_episode_steps: int = 1500  # 30 s at 50 Hz
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)

    @staticmethod
    def from_dict(data: dict) -> "EnvConfig":
        """Build from a parsed mapping; unknown keys raise ``ValueError``."""
        data = dict(data)
        spawn = _from_mapping(SpawnConfig, data.pop("spawn", {}))
        reward = _from_mapping(RewardConfig, data.pop("reward", {}))
        termination = _from_mapping(TerminationConfig, data.pop("termination", {}))
        cfg = _from_mapping(EnvConfig, data | {"spawn": {}, "reward": {}, "termination": {}})
        cfg.spawn, cfg.reward, cfg.termination = spawn, reward, termination
        return cfg


@dataclass
class TrainConfig:
    """SAC hyperparameters for one stage (defaults tuned for 50 Hz, ~1500-step episodes)."""

    total_steps: int = 200_000
    learning_rate: float = 3e-4
    buffer_size: int = 300_000
    batch_size: int = 256
    gamma: float = 0.995  # ~4 s credit horizon at 50 Hz: corner entry sees corner exit
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    learning_starts: int = 5_000  # a few episodes of random warmup before updates
    ent_coef: str | float = "auto"
    net_arch: list = field(default_factory=lambda: [256, 256])
    eval_freq: int = 10_000  # steps between eval-callback runs (0 disables)
    n_eval_episodes: int = 5
    device: str = "auto"


@dataclass
class StageConfig:
    """One curriculum stage: name + environment + training hyperparameters."""

    name: str
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        """Plain-dict view for YAML."""
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "StageConfig":
        """Build from a parsed mapping; unknown keys raise ``ValueError``."""
        data = dict(data)
        env = EnvConfig.from_dict(data.pop("env", {}))
        train = _from_mapping(TrainConfig, data.pop("train", {}))
        cfg = _from_mapping(StageConfig, data | {"env": {}, "train": {}})
        cfg.env, cfg.train = env, train
        return cfg

    def to_yaml(self, path) -> None:
        """Write the stage config to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @staticmethod
    def from_yaml(path) -> "StageConfig":
        """Read a stage config from a YAML file."""
        with open(path) as f:
            return StageConfig.from_dict(yaml.safe_load(f))

    @staticmethod
    def load(name_or_path: str) -> "StageConfig":
        """Resolve a packaged stage name (``stage1_path_follow``) or an explicit YAML path."""
        path = Path(name_or_path)
        if path.suffix in (".yaml", ".yml") or path.exists():
            return StageConfig.from_yaml(path)
        resource = importlib.resources.files("drift_core.environments") / "configs"
        candidate = resource / f"{name_or_path}.yaml"
        if not candidate.is_file():
            available = sorted(p.stem for p in candidate.parent.iterdir() if p.suffix == ".yaml")
            raise FileNotFoundError(
                f"Unknown stage config {name_or_path!r}; packaged stages: {available}"
            )
        return StageConfig.from_dict(yaml.safe_load(candidate.read_text()))


def list_stages() -> list[str]:
    """Names of the packaged curriculum stage configs."""
    resource = importlib.resources.files("drift_core.environments") / "configs"
    return sorted(p.stem for p in resource.iterdir() if p.suffix == ".yaml")
