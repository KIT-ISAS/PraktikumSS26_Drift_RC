# Black Formatter

Black is the Python formatter used for this project. It can be run with the command `black .` from the root of the repository. This will format all Python files in the repository according to the Black style guide. It is also part of the CI Pipeline, so any code that does not conform to the Black style will cause the CI checks to fail.

### VSCode Integration
If you are using VSCode in the dev container, the Black Formatter extension is installed automatically from `.devcontainer/devcontainer.json`. The workspace already sets Black as the default Python formatter and enables format-on-save for Python files.

### Collision with Flake8
Black and Flake8 can sometimes have conflicting style rules. To avoid this, we have configured Flake8 to allow double quotes, which is the default style used by Black. This configuration is set in the `ament_flake8.ini` file, which is part of the ROS 2 ament_flake8 package. See the `.devcontainer/Dockerfile` for details on how this is set up in the development environment.