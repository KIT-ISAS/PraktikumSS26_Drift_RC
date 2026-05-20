# Joystick Package
This package provides a ROS interface for controlling the car using a joystick controller (e.g. XBox). Its two main components are the `joy_node` and the `joystick_mapper_node`. 

### Joy Node
The `joy_node` is provided by the `joy` package of ROS2 (see: https://index.ros.org/p/joy/#jazzy). It reads the input from a joystick controller and publishes it as a `sensor_msgs/Joy` message on the `/joy` topic. The joystick in our case is mounted to `/dev/input/js0` and the `joy_node` is configured by default to read from this device (see `.devcontainer/docker-compose.yaml` where `/dev/input` is mounted). We don't have to do anything else to make it work, which is very convenient.

### Joystick Mapper Node
The `joystick_mapper_node` is a custom node that subscribes to the `/joy` topic that is published by the `joy_node` and maps the joystick inputs to control commands for the car. In our case we convert it to throttle and steering commands that are published on the `/car/vehicle_action_request` topic with a custom message. The mapping is defined in the `joystick_mapper_node` and can be customized to fit the specific joystick and control scheme you want to use by extending the abstract base class `JoystickMapper`. We currently use an XBox One controller.

### Docker Configuration
In order to access joystick devices from within the Docker container, we need to mount the `/dev/input` directory into the container. This is done in the `docker-compose.yaml` file. Additionally we need to ensure that the container has the necessary permissions to access the joystick device (device_cgroup_rules:
      - "c 13:* rmw").
To detect hot-plugging events, we need to mount `/run/udev` as well, which is also done in the `docker-compose.yaml` file. This allows the `joy_node` to detect when a joystick is connected or disconnected while the node is running.