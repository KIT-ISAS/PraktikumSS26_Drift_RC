"""Load a trained vehicle model from a run directory (or a bare ``model.pt``)."""

from pathlib import Path

from .base import VehicleModel
from .registry import get_model_class


def load_model_from_run(
    run_dir: str, device: str = "cpu", stochastic: bool = False
) -> VehicleModel:
    """
    Load the trained model saved in a run directory.

    ``run_dir`` may be a run directory containing ``model.pt`` or a direct path to
    a ``.pt`` checkpoint. The checkpoint must carry ``model_type``; the matching
    class's :meth:`VehicleModel.from_checkpoint` reconstructs the model.
    """
    import torch

    path = Path(run_dir)
    checkpoint_path = path / "model.pt" if path.is_dir() else path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

    # weights_only=True: our checkpoints are plain dicts of tensors + primitive
    # metadata (model_type, hidden_dim, normalization lists), so the restricted
    # unpickler suffices and avoids the arbitrary-code-execution risk of full pickle.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_type = checkpoint.get("model_type")
    if model_type is None:
        raise ValueError(f"Checkpoint {checkpoint_path} has no 'model_type' field")

    cls = get_model_class(model_type)
    return cls.from_checkpoint(checkpoint, device=device, stochastic=stochastic)
