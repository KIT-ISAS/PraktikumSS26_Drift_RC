FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    git \
    openssh-client \
    vim \
    nano \
    ros-jazzy-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

# configure flake8 to allow double quotes since the Black formatter uses them by default
RUN echo "inline-quotes = \"" >> /opt/ros/jazzy/lib/python3.12/site-packages/ament_flake8/configuration/ament_flake8.ini

WORKDIR /root/repo/drift_ws

# Create a user ALSA config that routes audio to a null device to silence
# ALSA errors when no sound hardware is present. Placed in /root since the
# devcontainer runs as root by default.
RUN cat > /root/.asoundrc <<'ASOUND'
pcm.null { type null }
pcm.!default { type plug; slave.pcm "null" }
ASOUND
RUN chmod 644 /root/.asoundrc || true

# Create virtual environment since Python doesn't allow system wide installation
RUN python3 -m venv /opt/venv

# Enable colored terminal output
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' ~/.bashrc

# Activate venv automatically
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install Python dependencies into venv
RUN pip install --no-cache-dir -r requirements.txt

# COPY src ./src

RUN source /opt/ros/jazzy/setup.bash && \
    colcon build

RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "source /root/repo/drift_ws/install/setup.bash" >> ~/.bashrc && \
    echo "source /opt/venv/bin/activate" >> ~/.bashrc

CMD ["bash"]