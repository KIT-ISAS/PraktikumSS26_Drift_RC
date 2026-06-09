#!/bin/bash
AMENT_FLAKE8_CONFIG=/opt/ros/jazzy/lib/python3.12/site-packages/ament_flake8/configuration/ament_flake8.ini
AMENT_PEP257_MAIN=/opt/ros/jazzy/lib/python3.12/site-packages/ament_pep257/main.py

# Ignore D401 "First line of Docstring should be in imperative mood"
sed -i 's/^extend-ignore = \(.*\)$/extend-ignore = \1,D401,I101/' $AMENT_FLAKE8_CONFIG
sed -i "/_ament_ignore = \[/,/\]/ s/\]/    'D401',\n]/" $AMENT_PEP257_MAIN
# configure flake8 to allow double quotes to avoid conflicts with other formatters (and double quotes are nicer)
echo 'inline-quotes = "' >> $AMENT_FLAKE8_CONFIG

# Add all packages in the workspace to the application-import-names configuration so that flake8 doesn't complain about imports from other packages in the workspace
# See README.md for more information about this workaround
PACKAGES=(example_pkg,holodeck_pkg,joystick_pkg,shared_interfaces,shared_pkg,track_pkg,vehicle_communication_pkg)
echo "application-import-names = $PACKAGES" >> $AMENT_FLAKE8_CONFIG