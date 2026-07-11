class Topics:
    """Topics used for communication between ROS nodes."""

    # ROS wide Topics
    JOY_NODE: str = "/joy"

    # Transforms
    TF_STATIC: str = "/tf_static"

    # Map
    MAP_BASE: str = "/map"
    LIGHTHOUSE_POSES: str = MAP_BASE + "/lighthouses"

    # Tracking source (libsurvive) raw outputs
    LIBSURVIVE_IMU: str = "/libsurvive/imu"

    # Tracking source (SteamVR) raw outputs
    STEAMVR_BASE: str = "/steamvr"
    STEAMVR_IMU: str = STEAMVR_BASE + "/imu"
    STEAMVR_POSE: str = STEAMVR_BASE + "/pose"

    # Track-related Topics
    TRACK_BASE: str = "/track"
    TRACK_BOUNDARY: str = TRACK_BASE + "/boundary"
    TRACK_CENTERLINE: str = TRACK_BASE + "/centerline"
    TRACK_VISUALIZATION: str = TRACK_BASE + "/visualization"
    TRACK_PROGRESS: str = TRACK_BASE + "/progress"
    TRACK_PROGRESS_VISUALIZATION: str = TRACK_BASE + "/progress_visualization"

    # Car-related Topics
    CAR_BASE: str = "/car"
    CAR_POSE: str = CAR_BASE + "/pose"
    CAR_ODOM: str = CAR_BASE + "/odom"
    CAR_IMU: str = CAR_BASE + "/imu"
    # Canonical car mesh marker — derived from CAR_POSE, so it stays absolute and
    # source-agnostic (mirrors the /track/* visualization topics) instead of
    # inheriting the localization source's node namespace.
    CAR_VISUALIZATION: str = CAR_BASE + "/visualization"
    VEHICLE_ACTION_REQUEST: str = CAR_BASE + "/vehicle_action_request"
    VEHICLE_STATUS: str = CAR_BASE + "/vehicle_status"

    # Agent control signals — gate whether the RL agent is allowed to drive.
    # The agent holds off until it receives a start signal and stops again on a
    # stop signal (std_msgs/Empty triggers).
    AGENT_BASE: str = "/agent"
    AGENT_START: str = AGENT_BASE + "/start"
    AGENT_STOP: str = AGENT_BASE + "/stop"
    # Emergency takeover: another controller (e.g. the joystick) has seized the
    # vehicle action topic. Unlike AGENT_STOP the agent must release control
    # immediately — no brake-to-standstill phase, which would fight the new
    # controller's commands.
    AGENT_OVERRIDE: str = AGENT_BASE + "/override"
