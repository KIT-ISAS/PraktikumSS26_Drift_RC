"""
Pure message->signal bridges shared by the agent and data-collection nodes.

These functions turn the canonical ROS messages (``/car/odom``, ``/track/progress``,
latched ``/track/centerline``) into the exact inputs the training environment feeds
its shared builders — the sim-to-real seam. They are ROS-message-aware but otherwise
pure (no ``self``, no node state), so both the deployment node (:mod:`.agent_node`)
and the recorder (:mod:`.data_collection_node`) share one implementation and it is
unit-testable against ``DriftEnv``.

- :func:`observation_from_messages` mirrors ``DriftEnv``'s observation.
- :func:`reward_from_messages` mirrors ``DriftEnv``'s per-step reward/termination.

The heavy lifting lives in ``vehicle_simulation`` (``build_observation``,
``compute_reward``); here we only extract the scalars each one needs.
"""

import math

from nav_msgs.msg import Odometry
import numpy as np

from shared_interfaces.msg import Track, TrackProgress
from vehicle_simulation.environments.observation import build_observation, curvature_lookahead
from vehicle_simulation.environments.rewards import (
    RewardConfig,
    StepOutcome,
    TerminationConfig,
    compute_reward,
)
from vehicle_simulation.transforms import wrap_angle, yaw_from_quaternion


def observation_from_messages(
    odom: Odometry,
    progress: TrackProgress,
    track: Track,
    prev_action: np.ndarray,
) -> np.ndarray:
    """
    Build the policy observation from the ROS messages (pure, unit-testable).

    Mirrors ``DriftEnv``'s observation exactly: every field maps 1:1 onto what
    the training environment feeds the shared builder.
    """
    car = odom.pose.pose.orientation
    tangent = progress.projected_pose.orientation
    heading = yaw_from_quaternion(car.x, car.y, car.z, car.w)
    track_heading = yaw_from_quaternion(tangent.x, tangent.y, tangent.z, tangent.w)
    curvatures_ahead = curvature_lookahead(
        track.curvature,
        track.cumulative_distance,
        progress.total_length,
        progress.distance,
    )
    return build_observation(
        lateral_error=progress.lateral_error,
        heading_error=heading - track_heading,
        vx=odom.twist.twist.linear.x,
        vy=odom.twist.twist.linear.y,
        yaw_rate=odom.twist.twist.angular.z,
        prev_action=prev_action,
        curvatures_ahead=curvatures_ahead,
    )


def _kappa_here(track: Track, progress: TrackProgress) -> float:
    """Signed local centerline curvature at the car's projection (matches DriftEnv)."""
    cumul = np.asarray(track.cumulative_distance, dtype=np.float64)
    curvature = np.asarray(track.curvature, dtype=np.float64)
    return float(
        np.interp(
            progress.distance,
            np.append(cumul, progress.total_length),
            np.append(curvature, curvature[0]),
        )
    )


def reward_from_messages(
    reward_cfg: RewardConfig,
    term_cfg: TerminationConfig,
    odom: Odometry,
    progress: TrackProgress,
    track: Track,
    action: np.ndarray,
    prev_action: np.ndarray,
    ds: float,
    dt: float,
    stalled: bool,
) -> StepOutcome:
    """
    Score one real-car step from the ROS messages (pure, unit-testable).

    Mirrors ``DriftEnv.step``'s reward/termination: derives ``speed``, ``sideslip``,
    ``heading_error`` and local curvature from the messages, then defers to the
    shared :func:`compute_reward`. The caller owns the wrap-aware progress ``ds``
    (via ``rewards.progress_delta``) and the stall-timer verdict ``stalled``, just
    as the environment does.
    """
    vx = odom.twist.twist.linear.x
    vy = odom.twist.twist.linear.y
    speed = math.hypot(vx, vy)
    sideslip = math.atan2(vy, vx) if (vx or vy) else 0.0

    car = odom.pose.pose.orientation
    tangent = progress.projected_pose.orientation
    heading = yaw_from_quaternion(car.x, car.y, car.z, car.w)
    track_heading = yaw_from_quaternion(tangent.x, tangent.y, tangent.z, tangent.w)
    heading_error = wrap_angle(heading - track_heading)

    return compute_reward(
        reward_cfg,
        term_cfg,
        ds=ds,
        speed=speed,
        lateral_error=progress.lateral_error,
        heading_error=heading_error,
        sideslip=sideslip,
        kappa_here=_kappa_here(track, progress),
        action=action,
        prev_action=prev_action,
        dt=dt,
        stalled=stalled,
    )
