"""
Pure message->signal bridge for the MPC node.

Mirrors :mod:`.observation_bridge` for the deterministic controller: extracts the
few scalars the lateral MPC and the speed controller need from the canonical ROS
messages, with no node state, so it is unit-testable in isolation.
"""

from nav_msgs.msg import Odometry

from drift_core.transforms import wrap_angle, yaw_from_quaternion
from shared_interfaces.msg import TrackProgress


def mpc_inputs_from_messages(
    odom: Odometry, progress: TrackProgress
) -> tuple[float, float, float, float]:
    """
    Return ``(lateral_error, heading_error, vx, s)`` for one MPC solve.

    ``lateral_error`` [m] is signed + = car left of the centerline,
    ``heading_error`` [rad] is the car heading minus the track-tangent heading
    (wrapped), ``vx`` [m/s] the body-frame forward speed, and ``s`` [m] the arc
    length along the centerline — all exactly as :class:`drift_core.agent.control.PathMpc`
    expects them.
    """
    car = odom.pose.pose.orientation
    tangent = progress.projected_pose.orientation
    heading = yaw_from_quaternion(car.x, car.y, car.z, car.w)
    track_heading = yaw_from_quaternion(tangent.x, tangent.y, tangent.z, tangent.w)
    return (
        float(progress.lateral_error),
        float(wrap_angle(heading - track_heading)),
        float(odom.twist.twist.linear.x),
        float(progress.distance),
    )
