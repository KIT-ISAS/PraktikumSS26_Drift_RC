# steamvr_pkg

SteamVR/OpenVR tracking source — the switchable alternative to the
libsurvive-based `holodeck_pkg`. It fulfils the tracking-source contract from
`car_localization_pkg`: publish TF `world -> tracker_link` plus a raw IMU
topic; everything downstream (`/car/pose`, `/car/odom`, track progress) is
source-agnostic.

```
HOST                                       CONTAINER (host network)
SteamVR ──openvr──> steamvr_bridge.py ──UDP JSON :15250──> steamvr_node
  (see steamvr_bridge/ at the repo root)                     ├─ TF  world -> steamvr_world           (static, calibration params)
                                                             ├─ TF  steamvr_world -> tracker_link    (dynamic)
                                                             ├─ /steamvr/imu    sensor_msgs/Imu      (pseudo-IMU, frame_id=tracker_link)
                                                             ├─ /steamvr/pose   geometry_msgs/PoseStamped (debug, world frame)
                                                             └─ /map/lighthouses geometry_msgs/PoseArray  (world frame, calibration aid)
```

SteamVR runs on the host (OpenVR needs local IPC to vrserver); the bridge
streams the **raw** OpenVR matrices/velocities and all coordinate conversion
happens here, unit-tested (`lib/conversion.py`: y-up OpenVR → z-up world).
The wire format is owned by `lib/protocol.py` (stdlib-only, imported by the
host script from the same checkout).

Note: SteamVR and libsurvive both claim the tracker dongle — only one source
can run at a time (`tracking_source` launch argument, see below).

## Run

```bash
# Host: SteamVR running, then
python3 steamvr_bridge/steamvr_bridge.py

# Container, standalone (first-bring-up milestone — tracker live in Foxglove):
ros2 launch steamvr_pkg steamvr_tracking.launch.py foxbridge:=true

# Container, full stack:
ros2 launch drift_ws/launch/manual_driving_with_viz.launch.py tracking_source:=steamvr
```

Sanity checks: `ros2 topic hz /steamvr/imu` (~100 Hz, the bridge poll rate)
and `ros2 run tf2_ros tf2_echo world tracker_link`.

## Pseudo-IMU

OpenVR's client API exposes pose + linear/angular velocity but no raw
accelerometer, so `/steamvr/imu` is synthesized: angular velocity from OpenVR
(rotated into the tracker body frame), linear acceleration finite-differenced
from OpenVR's velocity (+g up so it reads specific force at rest). It is
stamped `frame_id=tracker_link`, which `car_state_node` rotates into
`base_link` via TF — the libsurvive-specific `imu_link` chip transform is
bypassed entirely (the `tracker_link -> imu_link` static still exists but is
unused on this path). The finite-differenced acceleration is noisy: fine for
visualization/compatibility, not for training-grade data.

Caveat (flagged for bring-up): OpenVR's `vAngularVelocity` frame is
under-documented; the node assumes tracking-space, like `vVelocity`. Yaw the
tracker flat on the table — the rate must appear on the IMU body z axis with
positive sign for a counter-clockwise turn. If not, see the marked line in
`steamvr_node._publish_imu`.

## Calibration: `world -> steamvr_world`

The z-up conversion is fixed; what remains is a planar offset + yaw between
SteamVR's room-setup origin and the physical `world` frame — the parameters
`world_x/world_y/world_z/world_yaw_deg` in `config/steamvr_params.yaml`
(holodeck hardcodes its equivalent in `component.cpp`; here they are params).

1. **Coarse (lighthouse rectangle):** with the stack running, compare
   `/map/lighthouses` in Foxglove against the known physical positions
   (from holodeck's `component.cpp`: `(-2.70, ±2.33)` and `(2.80, ±2.33)` m).
   Tune `world_yaw_deg`, then `world_x`/`world_y`, until they land.
   `world_z` stays ≈ 0: the Standing universe is floor-calibrated.
2. **Fine (known car pose):** place the car at a marked pose (e.g. world
   origin, facing +x), read `/car/pose`, and trim the parameters.

Re-tune after redoing SteamVR room setup — the Standing-universe origin can
shift. `--universe raw` on the bridge is the uncalibrated fallback.

## Tracker body frame & mount

OpenVR's body convention for the Vive tracker was measured at bring-up
(static flat pose + sliding the tracker along its LED direction): **+y toward
the LED, -z out of the top face** — different from libsurvive's. The
conversion relabels it so `tracker_link` is always **z = top-face normal,
y = LED direction**; with the LED mounted toward the car front and the
tracker right-side up, the mount knobs for this source are
`tracker_yaw_offset_deg = -90`, `tracker_upside_down = false` (already set in
`manual_driving_with_viz.launch.py`).

Sanity checks after any mount change: tracker flat on the floor →
`tf2_echo steamvr_world tracker_link` shows roll ≈ pitch ≈ 0; push the car
forward → `/car/odom` `twist.linear.x` positive; `base_link` z-up in Foxglove.
