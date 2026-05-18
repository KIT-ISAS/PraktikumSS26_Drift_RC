class Topics:
    """Topics used for communication between nodes."""

    # ROS wide Topics
    JOY_NODE = "/joy"

    # Car-related Topics
    CAR_BASE = "/car"
    VEHICLE_ACTION_REQUEST = CAR_BASE + "/vehicle_action_request"
