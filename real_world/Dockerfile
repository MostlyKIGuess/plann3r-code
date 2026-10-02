# Robot-side image: ROS Noetic, RosAria (P3DX driver), RealSense driver, and the
# real_world/ scripts. Build from the repository root:
#   docker build -f real_world/Dockerfile -t plann3r-rrc .
# See docs/real-world.md.
FROM ros:noetic-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PLANN3R_REPO=/workspace/plann3r
ENV RRC_HOME=/opt/plann3r_rrc

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libaria-dev \
    python3-catkin-tools \
    python3-cv-bridge \
    python3-empy \
    python3-opencv \
    python3-pip \
    python3-rosdep \
    ros-noetic-diagnostic-updater \
    ros-noetic-geometry-msgs \
    ros-noetic-image-transport \
    ros-noetic-nav-msgs \
    ros-noetic-realsense2-camera \
    ros-noetic-realsense2-description \
    ros-noetic-roslaunch \
    ros-noetic-sensor-msgs \
    ros-noetic-tf \
    ros-noetic-tf2-ros \
    tmux \
    udev \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# RosAria is not shipped in the default Noetic apt repos, so build it from source.
RUN mkdir -p /root/catkin_ws/src && \
    cd /root/catkin_ws/src && \
    git clone --depth=1 https://github.com/amor-ros-pkg/rosaria.git && \
    cd /root/catkin_ws && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make -DCMAKE_BUILD_TYPE=Release"

RUN pip3 install --no-cache-dir \
    numpy \
    pillow \
    requests

# Install pixi so the same image can launch repo-side tasks when the repo is
# mounted at /workspace/plann3r. Pixi creates/uses the project environment at
# runtime; the image itself stays lightweight enough for robot-side ROS use.
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

RUN printf '%s\n' \
    'source /opt/ros/noetic/setup.zsh' \
    'source /root/catkin_ws/devel/setup.zsh' \
    > /root/.rosrc

RUN printf '%s\n' \
    'source /root/.rosrc' \
    'export PLANN3R_REPO=/workspace/plann3r' \
    'cd "${PLANN3R_REPO}" 2>/dev/null || true' \
    >> /root/.zshrc

WORKDIR ${RRC_HOME}
COPY real_world/record_realsense_map.py ${RRC_HOME}/record_realsense_map.py
COPY real_world/plann3r_ros_client.py ${RRC_HOME}/plann3r_ros_client.py
COPY real_world/plann3r_realworld_server.py ${RRC_HOME}/plann3r_realworld_server.py
COPY real_world/teleop ${RRC_HOME}/teleop
COPY real_world/teleop_joystick.py ${RRC_HOME}/teleop_joystick.py
COPY real_world/teleop_keyboard.py ${RRC_HOME}/teleop_keyboard.py
COPY real_world/build_plann3r_map.sh ${RRC_HOME}/build_plann3r_map.sh
COPY real_world/start_p3dx_realsense.sh ${RRC_HOME}/start_p3dx_realsense.sh
COPY real_world/start.sh ${RRC_HOME}/start.sh
RUN chmod +x ${RRC_HOME}/*.py ${RRC_HOME}/*.sh ${RRC_HOME}/teleop

WORKDIR ${PLANN3R_REPO}
ENTRYPOINT ["/opt/plann3r_rrc/start.sh"]
CMD ["zsh"]
