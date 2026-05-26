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

| Axis | Index | Mapped to |
|------|-------|-----------|
| Left stick horizontal | 0 | `steering` |
| Right trigger | 5 | `throttle` (forward) |
| Left trigger  | 2 | `throttle` (brake/reverse) |

Both triggers are combined into a single signed throttle value:
positive = forward (right trigger), negative = brake/reverse (left trigger).

### Gyro parameters

Fixed gyro settings are loaded from `config/joystick_params.yaml` and sent with every command:

```yaml
joystick_mapper:
  ros__parameters:
    gyro_mode: "soft"   # "soft" or "hard"
    gyro_gain: 0.0      # 0.0 (off) to 1.0 (full)
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
