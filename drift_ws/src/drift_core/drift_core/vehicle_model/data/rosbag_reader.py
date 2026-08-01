"""Read rosbag2 MCAP files and return aligned (state, action) DataFrames."""

from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from rosbags.typesys.store import Typestore

from ...transforms import yaw_from_quaternion
from ..models.esc import EscStateMachine


def _register_bag_types(typestore: Typestore, reader: Reader) -> None:
    """
    Register any message types the bag carries that aren't already known.

    Custom messages (e.g. ``shared_interfaces/msg/VehicleActionRequest``) are not
    in the default ROS 2 typestore, but rosbag2 stores their definition on each
    connection, so we register those on the fly. Keeps the core lib decoupled from
    the ``shared_interfaces`` package.
    """
    for connection in reader.connections:
        try:
            typestore.get_msgdef(connection.msgtype)
        except KeyError:
            msgdef = connection.msgdef
            if msgdef and msgdef.data:
                typestore.register(get_types_from_msg(msgdef.data, connection.msgtype))


def _replay_esc(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Replay one continuously-observed span through a fresh ESC, or None if it can't sync.

    The machine synchronizes on the first forward-throttle sample (forward throttle
    forces the lockout set from any state), so the prefix before it — where the
    recorded lockout state is unknowable — is trimmed. A span that never drives
    forward can never be synchronized and is dropped.
    """
    esc = EscStateMachine()
    forward = df["throttle"].to_numpy() >= esc.neutral_deadband
    if not forward.any():
        return None
    df = df.iloc[int(np.argmax(forward)) :].reset_index(drop=True)

    drive = np.empty(len(df))
    brake = np.empty(len(df))
    for i, (throttle, vx) in enumerate(zip(df["throttle"].to_numpy(), df["vx"].to_numpy())):
        cmd = esc.resolve(float(throttle), float(vx))
        drive[i] = cmd.drive
        brake[i] = cmd.brake
    return df.assign(drive=drive, brake=brake)


def _label_motor_commands(df: pd.DataFrame, max_gap_s: float) -> pd.DataFrame:
    """
    Replay the recorded pedal stream through the ESC to label ``(drive, brake)``.

    Raw negative throttle is ambiguous (brake or reverse, depending on command
    history), so ``derivative()`` training must use the ESC-*resolved* motor
    command. Feeding an :class:`EscStateMachine` the samples in time order, with the
    measured forward speed as the standstill input, reproduces what the real ESC did
    — a deterministic replay, not inference.

    The replay is restarted at every observation gap longer than ``max_gap_s``
    (tracking dropout, recording pause). The lockout flag is only reconstructible
    from *observed* commands and speeds: across a gap the car may well have rolled
    to a stop with a neutral pedal — the event that *clears* the lockout — and
    carrying the pre-gap flag over it would mislabel every subsequent negative pedal
    as brake when it was in fact reverse, destroying the very Markov property this
    labeling exists to provide. Each span therefore re-synchronizes from scratch, and
    a span that never drives forward is dropped rather than guessed at.
    """
    spans = [_replay_esc(span) for span in _split_on_gaps(df, max_gap_s)]
    kept = [span for span in spans if span is not None]
    if not kept:
        raise ValueError(
            "No continuously-observed span in the bag ever drives forward; the ESC "
            "replay cannot synchronize its lockout state"
        )
    return pd.concat(kept, ignore_index=True)


def _split_on_gaps(df: pd.DataFrame, max_gap_s: float) -> list[pd.DataFrame]:
    """Split rows into spans separated by observation gaps longer than ``max_gap_s``."""
    t_s = df["t_ns"].to_numpy() / 1e9
    cuts = np.where(np.diff(t_s) > max_gap_s)[0] + 1
    bounds = [0, *cuts.tolist(), len(df)]
    return [df.iloc[a:b].reset_index(drop=True) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def read_driving_data(bag_path: str | Path, max_gap_s: float = 1.0) -> pd.DataFrame:
    """
    Read a rosbag2 directory and return a time-aligned DataFrame.

    Reads the canonical car state from /car/odom (nav_msgs/Odometry: pose in
    world, twist already in the body frame — published by car_localization_pkg)
    and /car/vehicle_action_request (VehicleActionRequest). Velocities are taken
    straight from the recorded twist instead of re-differentiating the pose, so
    no consumer re-derives coordinate maths. Actions are forward-filled onto
    odom timestamps via merge_asof.

    The raw throttle pedal is additionally replayed through the ESC state machine
    (:func:`_label_motor_commands`) into the resolved motor command. The replay
    restarts at every observation gap longer than ``max_gap_s`` — keep this in sync
    with the ``max_gap_s`` used to segment episodes downstream — and samples before
    each span's first forward-throttle command are trimmed.

    Columns: t (s), x (m), y (m), heading (rad), vx (m/s, body forward),
    vy (m/s, body left), yaw_rate (rad/s), steering [-1,1], throttle [-1,1],
    drive [-1,1] (signed motor torque), brake [0,1] (mutually exclusive with drive),
    gyro_gain [0,1] (steering-stabilizer gyro strength commanded with the action)
    """
    bag_path = Path(bag_path)
    typestore = get_typestore(Stores.ROS2_JAZZY)

    odom_rows: list[tuple] = []
    action_rows: list[tuple] = []

    with Reader(bag_path) as reader:
        _register_bag_types(typestore, reader)
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == "/car/odom":
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                twist = msg.twist.twist
                heading = yaw_from_quaternion(
                    float(ori.x), float(ori.y), float(ori.z), float(ori.w)
                )
                odom_rows.append(
                    (
                        timestamp,
                        float(pos.x),
                        float(pos.y),
                        heading,
                        float(twist.linear.x),
                        float(twist.linear.y),
                        float(twist.angular.z),
                    )
                )

            elif connection.topic == "/car/vehicle_action_request":
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                action_rows.append(
                    (timestamp, float(msg.steering), float(msg.throttle), float(msg.gyro_gain))
                )

    if not odom_rows:
        raise ValueError(f"No /car/odom messages found in {bag_path}")
    if not action_rows:
        raise ValueError(f"No /car/vehicle_action_request messages found in {bag_path}")

    odom_df = (
        pd.DataFrame(odom_rows, columns=["t_ns", "x", "y", "heading", "vx", "vy", "yaw_rate"])
        .sort_values("t_ns")
        .reset_index(drop=True)
    )
    action_df = (
        pd.DataFrame(action_rows, columns=["t_ns", "steering", "throttle", "gyro_gain"])
        .sort_values("t_ns")
        .reset_index(drop=True)
    )

    # Forward-fill: each odom sample gets the most recent action before it
    merged = pd.merge_asof(odom_df, action_df, on="t_ns", direction="backward").dropna()

    merged = _label_motor_commands(merged, max_gap_s)

    t0 = merged["t_ns"].iloc[0]
    merged["t"] = (merged["t_ns"] - t0) / 1e9
    return merged.drop(columns=["t_ns"]).reset_index(drop=True)
