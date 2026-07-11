# steamvr_pkg

SteamVR/OpenVR tracking source — the switchable alternative to the
libsurvive-based `holodeck_pkg`. It fulfils the tracking-source contract from
`car_localization_pkg`: publish TF `world -> tracker_link` plus a raw IMU
topic; everything downstream (`/car/pose`, `/car/odom`, track progress) is
source-agnostic.

```
HOST                                       CONTAINER (host network)
SteamVR ──openvr──> steamvr_bridge.py ──UDP JSON :15250──> steamvr_node
  (see steamvr_bridge/ at the repo root)                     ├─ TF  steamvr_world -> tracker_link    (dynamic, raw tracker pose)
                                                             ├─ TF  steamvr_world -> <serial>        (static, one per lighthouse)
                                                             ├─ /steamvr/imu    sensor_msgs/Imu      (pseudo-IMU, frame_id=tracker_link)
                                                             └─ /steamvr/pose   geometry_msgs/PoseStamped (debug, steamvr_world frame)

world_anchor_node (car_localization_pkg, started by this package's launch):
  fits + latches TF world -> steamvr_world from the lighthouse frames and
  publishes /map/lighthouses (world frame).
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

## World frame: defined by the lighthouses, fitted automatically

The world frame is **defined by the lighthouse constellation** — identically
for both tracking sources. The reference file
`car_localization_pkg/config/lighthouses_world.yaml` states where each base
station sits in world coordinates (serials are the libsurvive IDs in hex, so
libsurvive and SteamVR refer to the same physical devices), and
`world_anchor_node` (car_localization_pkg, started by every tracking launch)
fits `world -> <source>_world` against it at startup. Consequences:

- **No routine calibration.** Redoing SteamVR room setup is harmless; the
  fit follows the universe origin wherever it lands each session.
- **A rigidly moved constellation moves the world (and track) with it** —
  the arena recenters automatically, for both sources.
- **A lighthouse moved relative to the others** shows up as fit residuals
  above `warn_threshold_m` (default 0.15 m) and a warning naming it. That
  means the reference file is stale — re-record it (below).

`ros2 run steamvr_pkg steamvr_calibrate` (no args) runs the same fit as a
diagnostic without touching anything (works for libsurvive too:
`--tracking-frame libsurvive_world`).

### Re-recording the reference (arena rearranged)

1. Bridge + `steamvr_node` running; place the tracker at the desired world
   origin (floor center) with its **LED pointing along the desired +x axis**.
2. ```bash
   ros2 run steamvr_pkg steamvr_calibrate --record-reference drift_ws/src/car_localization_pkg/config/lighthouses_world.yaml
   colcon build --packages-select car_localization_pkg   # launches read the INSTALLED copy!
   ```
3. Both sources pick the new frame up automatically — and remember the track
   YAMLs assume the world origin is where the track belongs.

### Manual mode

`world_anchor_node` with `world_calibration: manual` uses fixed `world_*`
parameters instead (solve them with `steamvr_calibrate --write <params
yaml>`, tracker at the floor center) — useful if the frame must not change
between sessions, e.g. for strictly comparable long-term recordings.

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
