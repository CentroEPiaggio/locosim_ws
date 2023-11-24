#!/bin/bash
set -e

# Set the default build type
BUILD_TYPE=RelWithDebInfo
colcon build  --symlink-install 