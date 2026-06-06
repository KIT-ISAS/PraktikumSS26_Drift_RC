# Foxglove Documentation
This document provides instructions on how to set up and use the Foxglove tool. It is a very powerful tool to visualize ROS data. You can download it [online](https://foxglove.dev/) and should get access to more features with your student email.


### Usage
To be able to use Foxglove with ROS2, you need to start the Foxglove bridge in your ROS2 workspace. You can do this by running the following command in your terminal:
```bash
ros2 run foxglove_bridge foxglove_bridge
```

But you can also start the bridge together with other nodes within a launch file, which is more convenient.
You then can visualize the data in Foxglove by connecting to the bridge. To do this, open Foxglove and click on "Open Connection" -> "Foxglove WebSocket". The URL is per default "ws://localhost:8765".


### Import Config
To visualize the data in Foxglove, you can import a configuration file. This file contains the settings for the visualization, such as which topics to subscribe to and how to display the data. You can find an example configuration file in the `foxglove/default_config.json` file. To import the configuration file, click on "Import Config" in Foxglove and select the file you want to import.  
You may need to install extensions in Foxglove to visualize the data properly. You can install these extensions by clicking on "Extensions" in Foxglove and searching for the extension you need (in our case it's the bar display extension). You can even create your own extensions if you want to visualize the data in a specific way. You can find more information about creating extensions in the [Foxglove Documentation](https://foxglove.dev/docs/extensions/overview).