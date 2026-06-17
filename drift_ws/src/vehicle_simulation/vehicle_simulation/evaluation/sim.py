"""
Load a trained model for the simulator (eval method 4).

Method 4 — driving the trained model in sim with a controller/keyboard — reuses
the existing ROS stack: the simulation_node already builds a model via
``build_model`` and the demo launch already wires keyboard/joystick + Foxglove.
This just exposes a CPU-loading helper and documents the launch command.
"""

from ..models.base import VehicleModel
from ..models.loader import load_model_from_run


def load_for_sim(run_dir: str, device: str = "cpu") -> VehicleModel:
    """Load a trained model from a run dir for real-time sim inference (CPU default)."""
    return load_model_from_run(run_dir, device=device)


def sim_launch_command(run_dir: str) -> str:
    """Return the ros2 launch command to drive this model in the simulator."""
    return (
        "ros2 launch simulation_pkg simulation_demo.launch.py "
        f"model_type:={run_dir}\n"
        "# then, in another terminal: ros2 run simulation_pkg keyboard_control\n"
        "# (or add joystick:=true). Connect Foxglove to ws://localhost:8765"
    )
