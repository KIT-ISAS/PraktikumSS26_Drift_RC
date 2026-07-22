"""
Write a recorded-vs-predicted MCAP for Foxglove (eval method 3), no ROS runtime.

Open-loop roll a model from a recording's initial state over its recorded actions,
then write both trajectories to an ``.mcap`` (via the ``rosbags`` writer) using
standard message types so Foxglove shows them with no ROS running:

  /eval/pose_recorded, /eval/pose_predicted   geometry_msgs/PoseStamped (animated)
  /eval/path_recorded, /eval/path_predicted   nav_msgs/Path            (overlay)
  /eval/error_position, /eval/error_heading    std_msgs/Float64        (plots)
  /eval/cmd_steering,  /eval/cmd_throttle       std_msgs/Float64        (plots)
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore

from ..models.base import VehicleAction, VehicleModel, VehicleState


def rollout_from_recording(
    model: VehicleModel,
    df: pd.DataFrame,
    *,
    start_idx: int = 0,
    horizon: int = 100,
    sanitize: bool = True,
):
    """
    Open-loop roll ``model`` from a recording's state over its recorded actions.

    ``df`` needs columns t, x, y, heading, vx, vy, yaw_rate, steering, throttle —
    exactly what ``read_driving_data`` returns from the recorded /car/odom.
    Returns ``(times, recorded_poses, predicted_poses, actions)`` with poses as
    ``(H+1, 3)`` arrays of ``[x, y, heading]``.

    With ``sanitize`` (the default) the segment is screened for localization jumps
    the same way the training pipeline is: the rollout is truncated at the first
    flagged transition so it never integrates across a tracker glitch, and seeding
    from a corrupted sample raises rather than silently rolling from a physically
    impossible state (an 8 m/s, 100 rad/s twist that makes the model "predict"
    nonsense and look broken when the *data* was the problem). Pass
    ``sanitize=False`` to roll over the raw samples regardless.
    """
    from ..data.sanity import detect_discontinuities

    # Match detect_discontinuities' own defaults so the seed check and the transition
    # check agree on what counts as a jump (rather than hardcoding one car's limits here).
    _SEED_MAX_SPEED, _SEED_MAX_YAW_RATE = 8.0, 25.0

    end = min(start_idx + horizon + 1, len(df))
    seg = df.iloc[start_idx:end].reset_index(drop=True)
    if len(seg) < 2:
        raise ValueError("Need at least 2 samples to roll out")

    if sanitize:
        # detect_discontinuities only validates the *transitions* and rows[1:], so the
        # seed row (row 0) — the state we actually start integrating from — is exempt.
        # Screen it directly against the same bounds: seeding a rollout from a corrupt
        # twist is the exact failure that made a good model look broken.
        seed_speed = float(np.hypot(seg["vx"][0], seg["vy"][0]))
        seed_yaw = abs(float(seg["yaw_rate"][0]))
        mask, _ = detect_discontinuities(seg)
        seeds_from_jump = seed_speed > _SEED_MAX_SPEED or seed_yaw > _SEED_MAX_YAW_RATE
        # mask is non-empty here (len(seg) >= 2), and mask[0] truthy implies mask.any().
        if mask[0] or seeds_from_jump:
            raise ValueError(
                f"start_idx={start_idx} seeds from a localization jump (recorded state "
                f"there is physically implausible: {seed_speed:.1f} m/s, {seed_yaw:.1f} "
                "rad/s). Pick a clean start, or pass sanitize=False to roll from it anyway."
            )
        if mask.any():
            # Row `first+1` carries the twist finite-differenced across the jump, so the
            # clean run ends at `first`.
            first = int(mask.argmax())
            seg = seg.iloc[: first + 1].reset_index(drop=True)
            if len(seg) < 2:
                raise ValueError(
                    f"start_idx={start_idx} has a localization jump at the very next "
                    "sample; no clean window to roll out. Pick a different start."
                )

    t = seg["t"].to_numpy()
    times = t - t[0]
    recorded = seg[["x", "y", "heading"]].to_numpy()
    actions = seg[["steering", "throttle"]].to_numpy()
    predicted = _roll_path(model, seg, actions, stochastic=False)
    return times, recorded, predicted, actions


def _roll_path(model, seg: pd.DataFrame, actions: np.ndarray, *, stochastic: bool):
    """
    One open-loop path over the segment's recorded actions; ``(H+1, 3)`` poses.

    Model-internal session state (latent actuators, ESC lockout) is reset first, so
    every path of a fan starts from the identical model state and only the sampled
    SDE noise separates them.
    """
    model._sim_latent = None
    model._esc = None
    t = seg["t"].to_numpy()
    state = VehicleState(
        x=float(seg["x"][0]),
        y=float(seg["y"][0]),
        heading=float(seg["heading"][0]),
        vx=float(seg["vx"][0]),
        vy=float(seg["vy"][0]),
        yaw_rate=float(seg["yaw_rate"][0]),
    )
    step = model.step_stochastic if stochastic else model.step
    path = [[state.x, state.y, state.heading]]
    for i in range(len(seg) - 1):
        dt = float(t[i + 1] - t[i])
        action = VehicleAction(steering=float(actions[i, 0]), throttle=float(actions[i, 1]))
        state = step(state, action, dt)
        path.append([state.x, state.y, state.heading])
    return np.array(path)


def sampled_rollout_fan(
    model: VehicleModel,
    df: pd.DataFrame,
    *,
    start_idx: int = 0,
    horizon: int = 100,
    samples: int = 10,
    sanitize: bool = True,
):
    """
    The stochastic counterpart of :func:`rollout_from_recording`.

    One mean path plus ``samples`` Euler–Maruyama paths drawn from the model's
    learned diffusion, all over the same recorded actions.

    Returns ``(times, recorded, predicted_mean, actions, sampled)`` where ``sampled``
    is a list of ``(H+1, 3)`` pose arrays. The spread of the fan against the recorded
    path is the visual check of the SDE's calibration: the recorded trajectory should
    look like *one plausible member* of the fan, not an outlier. The model must have
    been loaded with ``stochastic=True`` or the fan collapses onto the mean.
    """
    times, recorded, predicted, actions = rollout_from_recording(
        model, df, start_idx=start_idx, horizon=horizon, sanitize=sanitize
    )
    # Re-slice exactly the (possibly sanitize-truncated) window the mean path used.
    seg = df.iloc[start_idx : start_idx + len(times)].reset_index(drop=True)
    sampled = [_roll_path(model, seg, actions, stochastic=True) for _ in range(samples)]
    return times, recorded, predicted, actions, sampled


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def write_eval_mcap(
    out_path: str,
    times: np.ndarray,
    recorded_poses: np.ndarray,
    predicted_poses: np.ndarray,
    *,
    actions: np.ndarray | None = None,
    samples: list | None = None,
    frame_id: str = "world",
    prefix: str = "/eval",
) -> str:
    """
    Write recorded vs predicted trajectories to an MCAP; returns the path.

    ``samples`` (optional) is a list of ``(H+1, 3)`` sampled SDE paths (see
    :func:`sampled_rollout_fan`); each becomes ``{prefix}/path_sample_XX`` for a
    fan overlay in Foxglove, and ``{prefix}/sample_spread`` carries the fan's
    planar position std at each timestep for plotting against the actual error.
    """
    ts = get_typestore(Stores.ROS2_JAZZY)
    Time = ts.types["builtin_interfaces/msg/Time"]
    Header = ts.types["std_msgs/msg/Header"]
    Point = ts.types["geometry_msgs/msg/Point"]
    Quat = ts.types["geometry_msgs/msg/Quaternion"]
    Pose = ts.types["geometry_msgs/msg/Pose"]
    PoseStamped = ts.types["geometry_msgs/msg/PoseStamped"]
    PathMsg = ts.types["nav_msgs/msg/Path"]
    Float64 = ts.types["std_msgs/msg/Float64"]

    def stamp(t: float):
        return Time(sec=int(t), nanosec=int((t - int(t)) * 1e9))

    def pose_stamped(t, x, y, heading):
        half = heading / 2.0
        return PoseStamped(
            header=Header(stamp=stamp(t), frame_id=frame_id),
            pose=Pose(
                position=Point(x=float(x), y=float(y), z=0.0),
                orientation=Quat(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half)),
            ),
        )

    out = Path(out_path)
    writer = Writer(out, version=9, storage_plugin=StoragePlugin.MCAP)
    writer.open()
    try:
        conns = {
            name: writer.add_connection(f"{prefix}/{name}", msgtype, typestore=ts)
            for name, msgtype in (
                ("pose_recorded", "geometry_msgs/msg/PoseStamped"),
                ("pose_predicted", "geometry_msgs/msg/PoseStamped"),
                ("path_recorded", "nav_msgs/msg/Path"),
                ("path_predicted", "nav_msgs/msg/Path"),
                ("error_position", "std_msgs/msg/Float64"),
                ("error_heading", "std_msgs/msg/Float64"),
                ("cmd_steering", "std_msgs/msg/Float64"),
                ("cmd_throttle", "std_msgs/msg/Float64"),
            )
        }
        for k in range(len(samples or [])):
            name = f"path_sample_{k:02d}"
            conns[name] = writer.add_connection(
                f"{prefix}/{name}", "nav_msgs/msg/Path", typestore=ts
            )
        if samples:
            conns["sample_spread"] = writer.add_connection(
                f"{prefix}/sample_spread", "std_msgs/msg/Float64", typestore=ts
            )

        def w(conn, t, msg, msgtype):
            writer.write(conn, int(t * 1e9), ts.serialize_cdr(msg, msgtype))

        rec_path, pred_path = [], []
        for i, t in enumerate(times):
            rec = pose_stamped(t, recorded_poses[i, 0], recorded_poses[i, 1], recorded_poses[i, 2])
            pred = pose_stamped(
                t, predicted_poses[i, 0], predicted_poses[i, 1], predicted_poses[i, 2]
            )
            w(conns["pose_recorded"], t, rec, "geometry_msgs/msg/PoseStamped")
            w(conns["pose_predicted"], t, pred, "geometry_msgs/msg/PoseStamped")
            rec_path.append(rec)
            pred_path.append(pred)

            pos_err = float(np.hypot(*(predicted_poses[i, :2] - recorded_poses[i, :2])))
            head_err = abs(_wrap(float(predicted_poses[i, 2] - recorded_poses[i, 2])))
            w(conns["error_position"], t, Float64(data=pos_err), "std_msgs/msg/Float64")
            w(conns["error_heading"], t, Float64(data=head_err), "std_msgs/msg/Float64")
            if actions is not None and i < len(actions):
                steer = Float64(data=float(actions[i, 0]))
                throt = Float64(data=float(actions[i, 1]))
                w(conns["cmd_steering"], t, steer, "std_msgs/msg/Float64")
                w(conns["cmd_throttle"], t, throt, "std_msgs/msg/Float64")

        header = Header(stamp=stamp(float(times[0])), frame_id=frame_id)
        rec_msg = PathMsg(header=header, poses=rec_path)
        pred_msg = PathMsg(header=header, poses=pred_path)
        w(conns["path_recorded"], times[0], rec_msg, "nav_msgs/msg/Path")
        w(conns["path_predicted"], times[0], pred_msg, "nav_msgs/msg/Path")

        if samples:
            for k, sample in enumerate(samples):
                poses = [pose_stamped(t, *sample[i]) for i, t in enumerate(times)]
                msg = PathMsg(header=header, poses=poses)
                w(conns[f"path_sample_{k:02d}"], times[0], msg, "nav_msgs/msg/Path")
            xy = np.stack([s[:, :2] for s in samples])  # (K, H+1, 2)
            spread = np.sqrt(xy.var(axis=0).sum(axis=-1))  # planar std around the fan mean
            for i, t in enumerate(times):
                w(
                    conns["sample_spread"],
                    t,
                    Float64(data=float(spread[i])),
                    "std_msgs/msg/Float64",
                )
    finally:
        writer.close()
    return str(out)
