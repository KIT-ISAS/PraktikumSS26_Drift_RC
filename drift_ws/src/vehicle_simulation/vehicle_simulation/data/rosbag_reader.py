"""Read rosbag2 MCAP files and return aligned (pose, action) DataFrames."""

from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


def read_driving_data(bag_path: str | Path) -> pd.DataFrame:
    """
    Read a rosbag2 directory and return a time-aligned DataFrame.

    Reads /car/pose (PoseStamped) and /car/vehicle_action_request
    (VehicleActionRequest). Actions are forward-filled onto pose timestamps
    via merge_asof.

    Columns: t (s), x (m), y (m), heading (rad), steering [-1,1], throttle [-1,1]
    """
    bag_path = Path(bag_path)
    typestore = get_typestore(Stores.ROS2_JAZZY)

    pose_rows: list[tuple] = []
    action_rows: list[tuple] = []

    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == "/car/pose":
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                pos = msg.pose.position
                ori = msg.pose.orientation
                # Quaternion → yaw (rotation about Z only)
                heading = 2.0 * np.arctan2(float(ori.z), float(ori.w))
                pose_rows.append((timestamp, float(pos.x), float(pos.y), heading))

            elif connection.topic == "/car/vehicle_action_request":
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                action_rows.append((timestamp, float(msg.steering), float(msg.throttle)))

    if not pose_rows:
        raise ValueError(f"No /car/pose messages found in {bag_path}")
    if not action_rows:
        raise ValueError(f"No /car/vehicle_action_request messages found in {bag_path}")

    pose_df = (
        pd.DataFrame(pose_rows, columns=["t_ns", "x", "y", "heading"])
        .sort_values("t_ns")
        .reset_index(drop=True)
    )
    action_df = (
        pd.DataFrame(action_rows, columns=["t_ns", "steering", "throttle"])
        .sort_values("t_ns")
        .reset_index(drop=True)
    )

    # Forward-fill: each pose gets the most recent action before it
    merged = pd.merge_asof(pose_df, action_df, on="t_ns", direction="backward").dropna()

    t0 = merged["t_ns"].iloc[0]
    merged["t"] = (merged["t_ns"] - t0) / 1e9
    return merged.drop(columns=["t_ns"]).reset_index(drop=True)
