class Frames:
    """
    TF frame names used across the project (never hardcode these elsewhere).

    Coordinate convention is ISO 8855 everywhere: x forward, y left, z up, with
    yaw positive counter-clockwise (a left turn). The canonical frame graph is::

        world -> tracker_link -> base_link   (+ tracker_link -> imu_link)

    ``world`` is the single fixed, gravity-aligned frame shared by every car
    instance (real or simulated). ``tracker_link`` is whatever the tracking
    source reports (the only source-specific frame); ``base_link`` is the car
    body at the centre of gravity. The tracker->body transforms are static and
    derived from the measured car parameters, so swapping tracking systems only
    changes who publishes ``world -> tracker_link``.
    """

    # The shared fixed frame. Never namespaced per car instance.
    WORLD: str = "world"

    # Per-car frames (a sim instance prefixes these, e.g. "sim/base_link").
    BASE_LINK: str = "base_link"
    TRACKER_LINK: str = "tracker_link"
    IMU_LINK: str = "imu_link"
