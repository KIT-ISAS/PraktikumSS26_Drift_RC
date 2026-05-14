# ROS Documentation

### Links
- [ROS2 Documentation](https://docs.ros.org/en/jazzy/index.html)

### Useful Commands
- To source the ROS2 workspace (should be in docker .bashrc):
```bash
source /opt/ros/jazzy/setup.bash
source /root/repo/drift_ws/install/setup.bash
```

- To build the ROS2 workspace:
```bash
colcon build [--packages-select <package_name>]
```
- To run a ROS2 node:
```bash
ros2 run <package_name> <executable_name>
``` 
- To create a new package:
```bash
cd src
ros2 pkg create <package_name> --build-type ament_python
```