#!/bin/bash

# This file is run after the container is started through the post_start command in 
# docker-compose.yaml. It is used to set up the development environment, 
# such as configuring Flake8 and updating the list of packages in the workspace.

cd /root/repo/.devcontainer/scripts
source update_package_list.sh
source init_flake8.sh

# Install the ROS-free simulation/RL core library into the venv (editable).
# vehicle_simulation has no package.xml, so colcon never builds it; ROS nodes
# (track_pkg, simulation_pkg) import it from the venv via PYTHONPATH.
# Only the lightweight base is installed here. The heavy training extras
# (`[neural]` = torch, `[rl]` = gymnasium + stable-baselines3) are meant for the
# training / GPU machine: `pip install -e drift_ws/src/vehicle_simulation[neural,rl]`.
# A CPU-only deploy machine that just runs a trained agent uses the pinned
# `[rl-cpu]` extra instead — see the vehicle_simulation README for the one-liner.
/opt/venv/bin/pip install -e /root/repo/drift_ws/src/vehicle_simulation