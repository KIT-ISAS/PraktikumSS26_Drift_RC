#!/bin/bash

# This file is run after the container is started through the post_start command in 
# docker-compose.yaml. It is used to set up the development environment, 
# such as configuring Flake8 and updating the list of packages in the workspace.

cd /root/repo/.devcontainer/scripts
source update_package_list.sh
source init_flake8.sh