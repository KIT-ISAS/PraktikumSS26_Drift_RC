# car_localization_pkg

Source-agnostic car localization. Turns whatever a tracking system reports into
the **one canonical car state** — at the centre of gravity, in **ISO 8855**
(x forward, y left, z up, yaw positive counter-clockwise) — so the rest of the
system never depends on a specific tracking setup or re-derives coordinate maths.

## The contract a tracking source must satisfy

A source (libsurvive via `libsurvive_pkg`, or SteamVR via `steamvr_pkg`) only has
to provide, in its own gravity-aligned tracking frame (`libsurvive_world` /
`steamvr_world`):

1. **TF `<source>_world → tracker_link`** — the raw tracker pose. The pose may
   be "raw" (e.g. rotated); the mount transform below corrects it.
2. **Static TFs `<source>_world → <serial>`** for the lighthouses it sees
   (serial format `LHB-XXXXXXXX`). `world_anchor_node` (this package) fits and
   latches `world → <source>_world` against them, completing the chain — the
   lighthouse constellation *defines* the world frame, so both sources land in
   the same playground and the track moves with the lighthouses.
3. **A raw IMU** (`sensor_msgs/Imu`), if the tracker has one. `car_state_node`
   rotates it into `base_link` via TF from the message's `frame_id` — libsurvive
   stamps it `imu_link` (raw chip axes), steamvr stamps its pseudo-IMU
   `tracker_link` directly.

That is the entire boundary. Nothing downstream of `tracker_link` is
source-specific.

## What this package adds

```
world ──(source)──▶ tracker_link ──(mount_tf_node, static)──▶ base_link   [CoG, ISO 8855]
                    tracker_link ──(mount_tf_node, static)──▶ imu_link
```

- **`world_anchor_node`** — anchors `world → <source>_world` to the lighthouse
  constellation: fits the planar transform against
  `config/lighthouses_world.yaml` (`lighthouse_calibration.py`, pure/unit-
  tested), latches it on `/tf_static`, publishes `/map/lighthouses` in the
  world frame, and warns when a lighthouse no longer matches the reference
  (moved relative to the others → re-record the file, see steamvr_pkg's
  `steamvr_calibrate --record-reference`). It waits up to `settle_timeout_s`
  for the full reference constellation before the first fit, and if a changed
  lighthouse set later forces a re-fit, it warns with the resulting frame
  delta (downstream poses jump by that amount). Started by each tracking
  source's launch with the matching `tracking_frame`.
- **`mount_tf_node`** — publishes the static `tracker_link → base_link` (and
  `tracker_link → imu_link`) transforms, derived from the measured car
  parameters (`drift_core` `cars/*.yaml`). The mount orientation
  (tracker y-forward + inverted) is exposed as parameters
  (`tracker_yaw_offset_deg`, `tracker_upside_down`) to confirm against a known
  pose at bring-up.
- **`car_state_node`** — the single transform owner. Reads `world → base_link`
  from tf2, finite-differences the body-frame velocities, rotates the raw IMU
  into `base_link`, and publishes the **already-transformed** state:
  - `/car/pose` (`geometry_msgs/PoseStamped`) — pose only, in `world`.
  - `/car/odom` (`nav_msgs/Odometry`) — pose in `world` + body-frame twist.
  - `/car/imu` (`sensor_msgs/Imu`) — raw IMU rotated into `base_link`.
- **`car_viz_node`** — republishes a car pose as a coloured mesh marker.

## Running

This launch is the **single place that picks the tracking source.** It starts
the selected source, then derives the raw IMU topic and the tracker mount
convention (`tracker_yaw_offset_deg`, `tracker_upside_down`) from it — so the
tracker body frame, which differs per source, is handled in one place and the
parent launches only forward `tracking_source`.

```bash
ros2 launch car_localization_pkg car_localization.launch.py            # libsurvive (default)
ros2 launch car_localization_pkg car_localization.launch.py tracking_source:=steamvr
```

Real pipeline: included automatically by the top-level launches
(`manual_driving_with_viz.launch.py`, `agent_driving_with_viz.launch.py`, …),
which just pass `tracking_source:=libsurvive|steamvr` straight through.

### Real + simulated car together

Bring up a second instance namespaced and frame-prefixed so it coexists in the
shared `world` frame. A sim car provides its own tracker TF, so pass
`start_tracking_source:=false` to skip the source adapter:

```bash
ros2 launch car_localization_pkg car_localization.launch.py \
    namespace:=sim frame_prefix:=sim/ color_rgba:="[0.2, 0.6, 1.0, 1.0]" \
    start_tracking_source:=false
```

`world` is never prefixed; `base_link`/`tracker_link`/`imu_link` become
`sim/base_link` etc., and the topics move under `/sim`.

## Conventions

Frame names come from `shared_pkg.Frames`; topic names from `shared_pkg.Topics`.
Coordinate maths lives in `drift_core.transforms` (ROS-free). Do not
hardcode frame or topic strings, and do not re-implement transforms in consumers
— subscribe to the canonical `/car/*` topics instead.
