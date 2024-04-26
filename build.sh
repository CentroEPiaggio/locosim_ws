#!/bin/bash
set -e

# Set the default build type
BUILD_TYPE=RelWithDebInfo
source /opt/ros/humble/setup.bash
colcon build  --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release