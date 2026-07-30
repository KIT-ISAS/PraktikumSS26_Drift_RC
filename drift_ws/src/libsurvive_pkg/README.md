# libsurvive_pkg

This package is largely taken from https://github.com/asymingt/libsurvive_ros2#, a ROS2 wrapper for the [libsurvive](https://github.com/collabora/libsurvive) library, which provides support for the HTC Vive tracking system. The package includes ROS2 nodes and interfaces to interact with the Vive tracking hardware, allowing users to integrate Vive tracking data into their ROS2 applications. Then we don't have to use SteamVR which doesn't work well with Linux.

### Modifications
The original package has been modified slightly. The main changes include:
- Changed the package name to `libsurvive_pkg` to fit the context of our project.
- Added a patch for the libsurvive library to reduce stderr prints to have a clean build output.See `patches/stderr.patch` for details.
- Calibrated the tracking system and saved the config to `config/config.yaml`. Deactivated force recalibration in params. See next section for details.

### Configuration/Calibration
The libsurvive library automatically automatically calibrates the tracking system on startup. For calibration, you must have the Vive base stations set up and the Vive trackers connected via USB. The calibration process involves moving the tracker around in the tracking area and waiting at some poses for a moment to allow the system to determine their positions relative to the base stations. Once the tracker gets stationary for a moment, libsurvive takes this pose and adds it to its known poses. All these poses are used to optimize the localization of the base stations and the trackers. See the Readme of the [libsurvive package](https://github.com/collabora/libsurvive).

It saves its config file in `~/.config/libsurvive/config.yaml`. We have exported this file to `config/config.yaml` after calibrating it once. Now it gets copied into the `~/config/libsurvive` folder in the docker container on startup.  