"""Track geometry: Bezier centerlines, projection, and packaged track loading."""

from importlib.resources import files
from pathlib import Path

import yaml

from .bezier_track import BezierTrack
from .track_projector import Projection, TrackProjector

__all__ = ["BezierTrack", "TrackProjector", "Projection", "load_track", "list_tracks"]


def _tracks_dir():
    """Return the importlib traversable for the packaged ``tracks/`` directory."""
    return files("vehicle_simulation").joinpath("tracks")


def load_track(name_or_path: str) -> BezierTrack:
    """
    Load a track by packaged name (e.g. ``"figure_8"``) or by explicit YAML path.

    A value that ends in ``.yaml`` or points at an existing file is read directly;
    anything else is resolved against the packaged ``tracks/`` directory.
    """
    p = Path(name_or_path)
    if p.suffix == ".yaml" or p.exists():
        return BezierTrack.from_yaml(str(p))
    resource = _tracks_dir().joinpath(f"{name_or_path}.yaml")
    return BezierTrack.from_dict(yaml.safe_load(resource.read_text()))


def list_tracks() -> list[str]:
    """Return the names of all tracks bundled in the packaged ``tracks/`` directory."""
    names = [p.name for p in _tracks_dir().iterdir() if p.name.endswith(".yaml")]
    return sorted(n[: -len(".yaml")] for n in names)
