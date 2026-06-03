class Topics:
    """Topics used for communication between ROS nodes."""

    # ROS wide Topics
    JOY_NODE: str = "/joy"

    # Transforms
    TF_STATIC: str = "/tf_static"

    # Map
    MAP_BASE: str = "/map"
    LIGHTHOUSE_POSES: str = MAP_BASE + "/lighthouses"

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
    VEHICLE_ACTION_REQUEST: str = CAR_BASE + "/vehicle_action_request"
    VEHICLE_STATUS: str = CAR_BASE + "/vehicle_status"
