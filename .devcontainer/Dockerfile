FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color

SHELL ["/bin/bash", "-c"]

########## Installations ##########
RUN apt-get update && apt-get install -y \
    freeglut3-dev \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    python-is-python3 \
    git \
    git-lfs \
    openssh-client \
    vim \
    nano \
    ros-jazzy-foxglove-bridge \
    ros-jazzy-joy \
    libusb-1.0-0-dev \
    libpcap-dev \
    liblapacke-dev \
    libopenblas-dev \
    libx11-dev \
    zlib1g-dev \
    libsciplot-dev \
    gh \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Register the LFS smudge/clean filters system-wide (the repo tracks *.pptx via LFS)
RUN git lfs install --system --skip-repo

# The repo is bind-mounted with the host user's uid while the container runs as
# root; without this git refuses every command with "dubious ownership". The
# container is a single-user dev environment, so trusting all paths is fine.
RUN git config --system --add safe.directory '*'



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
# Make the ROS-free core lib importable by the system python3 that ROS nodes run
# under. The venv site-packages above is only *appended* to PYTHONPATH, so pip's
# editable (.pth-based) install is not picked up there; putting the package source
# on PYTHONPATH gives editable imports for both system and venv python.
ENV PYTHONPATH="$PYTHONPATH:/root/repo/drift_ws/src/vehicle_simulation"

RUN source /opt/ros/jazzy/setup.bash

# RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "source /root/repo/drift_ws/install/setup.bash" >> ~/.bashrc

# Persist shell history on the bash-history named volume (mounted at /commandhistory
# in docker-compose.yaml). `history -a` flushes after every command so history
# survives even an unclean container shutdown, not just a normal exit.
RUN mkdir -p /commandhistory && \
    echo "export HISTFILE=/commandhistory/.bash_history" >> ~/.bashrc && \
    echo "export HISTSIZE=10000" >> ~/.bashrc && \
    echo "export HISTFILESIZE=20000" >> ~/.bashrc && \
    echo 'export PROMPT_COMMAND="history -a; ${PROMPT_COMMAND:-}"' >> ~/.bashrc

COPY .devcontainer/scripts/.bash_aliases /root/.bash_aliases


WORKDIR /root/repo/drift_ws
CMD ["bash"]