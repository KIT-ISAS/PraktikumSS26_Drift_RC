# ROS2 Jazzy Documentation

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
cd ../../.devcontainerscripts
source update_package_list.sh
```
- To run tests:
```bash
colcon test --event-handlers console_direct+

```
- To run only the (non-)linter tests:
```bash
colcon test --event-handlers console_direct+ --pytest-args '-m not linter'
colcon test --event-handlers console_direct+ --pytest-args '-m linter'
```

# Ament Flake8
Ament Flake8 is the linter used for this project. It is part of the test suite and can be run with the command `colcon test --event-handlers console_direct+` from the root of the repository. This will run all tests, including the Flake8 linter tests that are generated automatically for each new package that is created with the `ros2 pkg create` command.
It is also part of the CI Pipeline, so any code that does not conform to the Flake8 style will cause the CI checks to fail.

### Configuration
The `main_with_errors(argv=[])` function that is used in the generated tests can take in a path to a config file with the `--config` flag. However, we would like to use the default configuration and not make any changes to generated files (as it may be forgotten when creating a new one). The default configuration is set in the `/opt/ros/jazzy/lib/python3.12/site-packages/ament_flake8/configuration/ament_flake8.ini` file. 
Flake8 prefers single quotes for strings, while the Ruff formatter uses double quotes by default. To avoid conflicts, we have configured Flake8 to allow double quotes in the `ament_flake8.ini` file. See the `.devcontainer/Dockerfile` for details on how this is set up in the development environment.  