FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color

SHELL ["/bin/bash", "-c"]

########## Installations ##########
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    python-is-python3 \
    git \
    openssh-client \
    vim \
    nano \
    ros-jazzy-foxglove-bridge \
    ros-jazzy-joy \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment since Ubuntu doesn't allow system wide installation
RUN python3 -m venv /opt/venv

# Install Python dependencies into venv
COPY requirements.txt .
RUN source /opt/venv/bin/activate && pip install -r requirements.txt


########## Custom System-wide Configurations ##########
# Enable colored terminal output
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' ~/.bashrc


########## Environment Variables ##########
# Activate venv automatically and make installed packages available for ROS
# see documentation/Python.md for details
ENV PATH="$PATH:/opt/venv/bin"
ENV PYTHONPATH="$PYTHONPATH:/opt/venv/lib/python3.12/site-packages" 

RUN source /opt/ros/jazzy/setup.bash

# RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "source /root/repo/drift_ws/install/setup.bash" >> ~/.bashrc

COPY .devcontainer/scripts/.bash_aliases /root/.bash_aliases


WORKDIR /root/repo/drift_ws
CMD ["bash"]