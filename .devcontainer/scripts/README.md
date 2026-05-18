This folder is for scripts that are used in the devcontainer setup. 
These scripts can be used to automate tasks such as installing dependencies, setting up the environment, or running tests.


Upon startup, the `post_start.sh` script is executed (see `docker-compose.yaml`). It calls the `update_package_list.sh` script, which updates the list of packages in the workspace in the Flake8 and Ruff configurations. This ensures that the linting tools are aware of all the packages in the workspace and can properly lint the code. It's a weird workaround but necessary for automation. It cannot be put in the dockerfile because you need to have the workspace mounted to get all packages with colcon list.