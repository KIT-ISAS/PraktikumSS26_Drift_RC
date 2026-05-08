FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    git \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /drift_ws

# Create virtual environment since Python doesn't allow system wide installation
RUN python3 -m venv /opt/venv

# Enable colored terminal output
RUN echo "export PS1='\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc \
    && echo "alias ls='ls --color=auto'" >> ~/.bashrc \
    && echo "alias grep='grep --color=auto'" >> ~/.bashrc

# Activate venv automatically
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install Python dependencies into venv
RUN pip install --no-cache-dir -r requirements.txt

# COPY src ./src

RUN source /opt/ros/jazzy/setup.bash && \
    colcon build

RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "source /drift_ws/install/setup.bash" >> ~/.bashrc && \
    echo "source /opt/venv/bin/activate" >> ~/.bashrc

CMD ["bash"]