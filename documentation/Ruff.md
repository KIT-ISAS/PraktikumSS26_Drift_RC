# Ruff

Ruff is the Python linter and formatter used for this project. It can be run with the command `ruff check .` from the root of the repository. This will check all Python files in the repository according to the Ruff style guide. It is also part of the CI Pipeline, so any code that does not conform to the Ruff style will cause the CI checks to fail.

### Configuration
The Ruff configuration for this project is defined in the `pyproject.toml` file at the root of the repository. This file specifies the rules and settings that Ruff will use when checking the code. The rules were set according to the ROS2 style guide (pep8/pep257) that are part of the auto-generated package tests.

 The list of first-party packages defined in `pyproject.toml` is dynamically updated based on the packages present in the workspace, using the `.devcontainer/scripts/update_package_list.sh` script which is automatically executed on container startup.

### VSCode Integration
If you are using VSCode in the dev container, the Ruff extension is installed automatically from `.devcontainer/devcontainer.json`. The workspace already sets Ruff as the default Python formatter and enables format-on-save and "fix-on-save" for Python files (see the devcontainer.json).

### Collision with Flake8
Ruff and Flake8 can sometimes have conflicting style rules. To avoid this, we have configured Flake8 to allow double quotes, which is the default style used by Ruff. This configuration is set in the `ament_flake8.ini` file, which is part of the ROS 2 ament_flake8 package. See the `.devcontainer/scripts/init_flake8.sh` for details on how this is set up in the development environment.

