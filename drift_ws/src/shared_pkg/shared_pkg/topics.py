class Topics:
    """Topics used for communication between ROS nodes."""

    # ROS wide Topics
    JOY_NODE: str = "/joy"

    # Track-related Topics
    TRACK_BOUNDARY: str = "/track/boundary"

    # Car-related Topics
    CAR_BASE: str = "/car"
    VEHICLE_ACTION_REQUEST: str = CAR_BASE + "/vehicle_action_request"
    VEHICLE_STATUS: str = CAR_BASE + "/vehicle_status"
