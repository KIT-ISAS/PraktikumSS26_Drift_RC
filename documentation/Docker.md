# Docker Documentation #
This document provides instructions on how to set up and use the Docker environment for the Drift RC project. The Docker setup allows you to run the ROS2 workspace in a containerized environment, ensuring consistency across different development setups.

### Dockerfile
The `Dockerfile` defines the environment for the ROS2 workspace. It is based on the official ROS2 Jazzy image and includes additional configurations for Python virtual environments and terminal settings.

### docker-compose.yaml
The `docker-compose.yaml` file defines the services for the Docker environment. It specifies the build context, container name, volume mounts, and network settings for the ROS2 workspace.

### requirements.txt
The `requirements.txt` file lists the Python dependencies required for the project. They are installed when the docker is built into a virtual environment within the Docker container to avoid conflicts with system-wide packages.

### Building and Running the Docker Container
To build and run the Docker container, follow these steps:
1. Ensure you have Docker and Docker Compose installed on your system.
2. Navigate to the project directory containing the `Dockerfile` and `docker-compose.yaml`.
3. Build the Docker image using the command:
```bash
docker-compose build
```
4. Start the Docker container using the command:
```bash
docker-compose up -d
```
5. To access the container's terminal, use the command:
```bash
docker exec -it ros2_drift_ws_dev bash
```
### VsCode Remote Development
You can also use Visual Studio Code's Remote Development extension to work directly within the Docker container. To set this up:
1. Install the Remote Development extension in Visual Studio Code.
2. Open the project folder in Visual Studio Code.
3. Open the Command Palette (Ctrl+Shift+P) and select "Remote-Containers: Open Folder in Container".


Keep in mind that you need to Rebuild the container if you make changes to the `Dockerfile`, `docker-compose.yaml`, or `requirements.txt` to ensure that the changes are reflected in the development environment.