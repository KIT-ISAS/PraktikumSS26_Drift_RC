class Topics:
    """Topics used for communication between nodes."""

    # ROS wide Topics
    JOY_NODE: str = "/joy"

    # Car-related Topics
    CAR_BASE: str = "/car"
    VEHICLE_ACTION_REQUEST: str = CAR_BASE + "/vehicle_action_request"
