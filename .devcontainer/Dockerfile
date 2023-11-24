FROM osrf/ros:humble-desktop-full

# 
ENV DEBIAN_FRONTEND=noninteractive
# RUN curl -sSL http://get.gazebosim.org | sh

# RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list' \
#     && wget https://packages.osrfoundation.org/gazebo.key -O - | apt-key add - \

# gazebo11 libgazebo11-dev

RUN apt-get update \
    && apt-get install -y ros-humble-gazebo-ros-pkgs \
    ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-gazebo-ros2-control \
    ros-humble-rqt-robot-steering \
    tmux python3-pip\
    ros-humble-joint-state-publisher-gui \ 
    ros-humble-plotjuggler-ros \
    wget \
    chrony \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*


ENV DEBIAN_FRONTEND=dialog

# Create a new user
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && apt-get update \
    && apt-get install -y sudo \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

ENV HOME /home/${USERNAME} 


# Enable bash history so it can be mounted and become persistent
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory \
    && echo "${SNIPPET}" >> "/home/$USERNAME/.bashrc"

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER ${USERNAME}

# RUN mkdir ${HOME}/init 
# COPY .devcontainer/requirements.txt ${HOME}/init/requirements

RUN pip -q install pip --upgrade \
    && pip3 install rl_games torch pymeshlab

# Set up auto-source of workspace for ros user
ARG WORKSPACE
RUN echo "source /opt/ros/humble/setup.bash" >> /home/ros/.bashrc
RUN echo "if [ -f ${WORKSPACE}/install/setup.bash ]; then source ${WORKSPACE}/install/setup.bash; fi" >> /home/ros/.bashrc
