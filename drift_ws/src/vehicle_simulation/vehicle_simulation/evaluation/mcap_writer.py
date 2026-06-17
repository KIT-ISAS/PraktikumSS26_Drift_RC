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
    model: VehicleModel, df: pd.DataFrame, *, start_idx: int = 0, horizon: int = 100
):
    """
    Open-loop roll ``model`` from a recording's state over its recorded actions.

    ``df`` needs columns t, x, y, heading, vx, vy, yaw_rate, steering, throttle —
    exactly what ``read_driving_data`` returns from the recorded /car/odom.
    Returns ``(times, recorded_poses, predicted_poses, actions)`` with poses as
    ``(H+1, 3)`` arrays of ``[x, y, heading]``.
    """
    end = min(start_idx + horizon + 1, len(df))
    seg = df.iloc[start_idx:end].reset_index(drop=True)
    if len(seg) < 2:
        raise ValueError("Need at least 2 samples to roll out")

    t = seg["t"].to_numpy()
    times = t - t[0]
    recorded = seg[["x", "y", "heading"]].to_numpy()
    actions = seg[["steering", "throttle"]].to_numpy()

    state = VehicleState(
        x=float(seg["x"][0]),
        y=float(seg["y"][0]),
        heading=float(seg["heading"][0]),
        vx=float(seg["vx"][0]),
        vy=float(seg["vy"][0]),
        yaw_rate=float(seg["yaw_rate"][0]),
    )
    predicted = [[state.x, state.y, state.heading]]
    for i in range(len(seg) - 1):
        dt = float(t[i + 1] - t[i])
        action = VehicleAction(steering=float(actions[i, 0]), throttle=float(actions[i, 1]))
        state = model.step(state, action, dt)
        predicted.append([state.x, state.y, state.heading])

    return times, recorded, np.array(predicted), actions


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def write_eval_mcap(
    out_path: str,
    times: np.ndarray,
    recorded_poses: np.ndarray,
    predicted_poses: np.ndarray,
    *,
    actions: np.ndarray | None = None,
    frame_id: str = "world",
    prefix: str = "/eval",
) -> str:
    """Write recorded vs predicted trajectories to an MCAP; returns the path."""
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
    finally:
        writer.close()
    return str(out)
