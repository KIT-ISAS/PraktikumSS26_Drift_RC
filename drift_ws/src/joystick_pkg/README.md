# Joystick Package

Provides a ROS 2 interface for controlling the car with a joystick (e.g. Xbox One).
Two nodes work together: `joy_node` reads raw hardware input, `joystick_mapper_node`
converts it to car commands.

## Joy Node

Provided by the ROS 2 `joy` package. Reads from `/dev/input/js0` and publishes
`sensor_msgs/Joy` on `/joy`. The device is mounted into the Docker container via
`docker-compose.yaml` — no extra setup needed. Hot-plugging is supported via `/run/udev`.

## Joystick Mapper Node

Subscribes to `/joy` and publishes `shared_interfaces/VehicleActionRequest` on
`/car/vehicle_action_request`.

Default axis mapping (configurable in `config/joystick_params.yaml`):

| Input | Axis/Index | Mapped to |
|-------|------------|-----------|
| Left stick horizontal | axis 0 | `steering` |
| Right trigger | axis 5 | `throttle` (forward) |
| Left trigger  | axis 2 | `throttle` (brake/reverse) |
| D-pad up/down | axis 7 | gyro gain (+/− `gyro_gain_step`) |
| D-pad left/right | axis 6 | gyro mode (soft / hard) |

Both triggers are combined into a single signed throttle value:
positive = forward (right trigger), negative = brake/reverse (left trigger).

The d-pad is exposed by the driver as hat axes (not buttons), so the node uses
rising-edge detection to fire once per press regardless of how long it is held.

### Gyro parameters

Gyro settings start from the values in `config/joystick_params.yaml` and can be
adjusted at runtime via the d-pad:

| D-pad | Effect |
|-------|--------|
| Up | increase `gyro_gain` by `gyro_gain_step` (max 1.0) |
| Down | decrease `gyro_gain` by `gyro_gain_step` (min 0.0) |
| Left | set `gyro_mode` to `"soft"` |
| Right | set `gyro_mode` to `"hard"` |

```yaml
joystick_mapper:
  ros__parameters:
    gyro_mode: "soft"      # "soft" or "hard"
    gyro_gain: 1.0         # 0.0 (off) to 1.0 (full)
    gyro_gain_step: 0.1    # increment/decrement per d-pad up/down press
```

### Deadzone

```yaml
joy_node:
  ros__parameters:
    deadzone: 0.01
```

Axis values below this threshold are treated as zero.

## Extending the mapper

To support a different controller, subclass `JoystickMapper` (in
`joystick_pkg/lib/joystick_mapper.py`) and implement the axis extraction methods.
The current implementation (`XBoxOneMapper`, in `xbox_one_mapper.py`) handles
Xbox One trigger normalization (triggers report +1 at rest, −1 at full press)
and treats untouched triggers (reported as 0 by `joy_node` until first moved)
as "released" so the car doesn't get a phantom half-throttle at startup.

## Docker configuration

- `/dev/input` is mounted for joystick access
- `device_cgroup_rules: ["c 13:* rmw"]` grants permission
- `/run/udev` is mounted for hot-plug detection
