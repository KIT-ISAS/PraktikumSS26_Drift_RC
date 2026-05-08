# Foxglove Documentation
This document provides instructions on how to set up and use the Foxglove tool. It is a very powerful tool to visualize ROS data. You can download it [online](https://foxglove.dev/) and should get access to more features with your student email.


### Usage
To be able to use Foxglove with ROS2, you need to start the Foxglove bridge in your ROS2 workspace. You can do this by running the following command in your terminal:
```bash
ros2 run ros_jazzy_foxglove_bridge ros_jazzy_foxglove_bridge
```

But you can also start the bridge together with other nodes within a launch file, which is more convenient.
You then can visualize the data in Foxglove by connecting to the bridge. To do this, open Foxglove and click on "Open Connection" -> "Foxglove WebSocket". The URL is per default "ws://localhost:8765".


```