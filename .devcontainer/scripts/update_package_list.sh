#!/bin/bash

# This script updates the list of packages in the workspace in the Flake8 and Ruff configurations. 

(
    cd /root/repo/drift_ws
    # ROS packages come from colcon; pure-Python libs without a package.xml
    # (e.g. vehicle_simulation) are invisible to colcon, so append them here.
    EXTRA_PACKAGES="vehicle_simulation"
    NAMES=$( { colcon list --names-only; echo "$EXTRA_PACKAGES"; } | sort -u )
    PACKAGES=$(echo "$NAMES" | paste -sd ',' --)
    PACKAGES_WITH_QUOTES=$(echo "$NAMES" | sed 's/^/"/; s/$/"/' | paste -sd ',' --)

    # Update Flake8 configuration to include the packages in the workspace as known first-party packages

    sed -i "s/^PACKAGES=(.*)/PACKAGES=($PACKAGES)/" ../.devcontainer/scripts/init_flake8.sh

    # Update Ruffs configuration to include the packages in the workspace as known first-party packages
    sed -i "s/\(known-first-party\s*=\s*\)[[].*[]]/\1[$PACKAGES_WITH_QUOTES]/" ../pyproject.toml
    echo "Updated package list: $PACKAGES"
)
