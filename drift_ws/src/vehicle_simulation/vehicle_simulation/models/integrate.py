"""
Pose integration and trajectory rollout shared by inference and training.

:func:`integrate_pose` is duck-typed: it works on plain Python/numpy scalars and
arrays (inference) and on torch tensors (differentiable training rollouts) without
importing torch at module load, preserving the torch-free light path.
"""

import numpy as np

from .base import VehicleAction, VehicleModel, VehicleState


def _is_tensor(x) -> bool:
    """True if x is a torch tensor, without importing torch eagerly."""
    return type(x).__module__.split(".", 1)[0] == "torch"


def dt_column(dt, ref):
    """
    Broadcast a scalar or ``(B,)`` dt to multiply a ``(B, D)`` derivative tensor.

    ``ref`` supplies device/dtype. Returns a scalar tensor for scalar ``dt`` or a
    ``(B, 1)`` tensor for a batched ``dt``.
    """
    import torch

    if not torch.is_tensor(dt):
        return torch.as_tensor(float(dt), dtype=ref.dtype, device=ref.device)
    dt = dt.to(ref.device, ref.dtype)
    return dt.reshape(-1, 1) if dt.ndim >= 1 else dt


def integrate_pose(pose, vx, vy, yaw_rate, dt, method: str = "midpoint"):
    """
    Integrate a world-frame pose ``[x, y, heading]`` by one step.

    ``pose`` has last dimension 3 (``[x, y, heading]``); ``vx``/``vy`` are
    body-frame velocities, ``yaw_rate`` rad/s, ``dt`` seconds (scalar or batched).
    Works for numpy arrays and torch tensors. ``method`` is ``"midpoint"``
    (heading at the half-step, smoother) or ``"euler"``.
    """
    if _is_tensor(pose):
        import torch

        x, y, heading = pose[..., 0], pose[..., 1], pose[..., 2]
        ref = heading if method == "euler" else heading + 0.5 * yaw_rate * dt
        cos_h, sin_h = torch.cos(ref), torch.sin(ref)
        x_new = x + (vx * cos_h - vy * sin_h) * dt
        y_new = y + (vx * sin_h + vy * cos_h) * dt
        return torch.stack([x_new, y_new, heading + yaw_rate * dt], dim=-1)

    pose = np.asarray(pose, dtype=np.float64)
    x, y, heading = pose[..., 0], pose[..., 1], pose[..., 2]
    ref = heading if method == "euler" else heading + 0.5 * yaw_rate * dt
    cos_h, sin_h = np.cos(ref), np.sin(ref)
    x_new = x + (vx * cos_h - vy * sin_h) * dt
    y_new = y + (vx * sin_h + vy * cos_h) * dt
    return np.stack([x_new, y_new, heading + yaw_rate * dt], axis=-1)


def simulate(
    model: VehicleModel,
    state0: VehicleState,
    actions: np.ndarray,
    dts: np.ndarray | float,
    *,
    sample: bool = False,
) -> list[VehicleState]:
    """
    Roll a model forward through an action sequence (numpy, via ``step``).

    ``actions`` is ``(H, 2)`` of ``[steering, throttle]``; ``dts`` is a scalar or
    ``(H,)`` per-step timestep. Returns the ``H + 1`` states starting at
    ``state0``. Used by evaluation / MCAP export; works for every model.
    """
    actions = np.asarray(actions, dtype=np.float64).reshape(-1, 2)
    horizon = len(actions)
    if np.isscalar(dts):
        dts = np.full(horizon, float(dts))
    else:
        dts = np.asarray(dts, dtype=np.float64).reshape(-1)

    states = [state0]
    state = state0
    for i in range(horizon):
        action = VehicleAction(steering=float(actions[i, 0]), throttle=float(actions[i, 1]))
        dt = float(dts[i])
        if sample:
            state = model.step_stochastic(state, action, dt)
        else:
            state = model.step(state, action, dt)
        states.append(state)
    return states
