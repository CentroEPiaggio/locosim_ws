## Locomotion Simulator Workspace

A complete simulation environment for the 8DOF quadrupedal robot.

This repository contains the following packages:


- `mulinex_description`: URDF description of the robot.
- `mulinex_gazebo`: Gazebo simulation of the robot.
- `pi3hat_moteus_int_msgs`: Messages for the communication between the controller and the simulation.
- `rbt_pd_cnt`: PD controller for the robot.
- `rlg_quad_controller`: Run that perform inference of a trained neural network and send the commands to the robot.
- `locomotion_experiment`: Launch files and nodes to generate references command for the controller, useful when running experiments.
  

## Installation

Clone the repo:

        git clone --recursive https://github.com/CentroEPiaggio/locosim_ws

This is a project built on devcontainer. To install it, you need to have docker and vscode installed with the extension `ms-vscode-remote.remote-containers`.
 Then, you can open the project in vscode and click on the button on the bottom left corner and select "**Reopen in container**" This will build the devcontainer and install all the dependencies.


> **WARNING:** The project is configured to run with NVIDIA GPU access. If you don't have an NVIDIA GPU, you need to change the `devcontainer.json` and remove the `--gpus=all` flag from the  run args.
By doing this you will not also be able to run the neural network controller.

### Host installation
If you have ros humble installed on your host, you can run the simulation without the devcontainer, just open the workspace and build it with colcon.

## Usage
**Run all the commands from the terminal inside vscode**

First, you need to build the project. To do so, you can run the build task of VScode with

        CTRL + SHIFT + B

or, in a terminal:

        colcon build --symlink-install

This will build the packages and create a symlink to the install folder.

To run the simulation, you can use 

        ros2 launch mulinex_gazebo launch sim.launch.py
        
This will launch the simulation and the PD controller.
To run the neural network controller, you can use `ros2 launch rlg_quad_controller mulinex_simulation.launch.py`. This will launch the simulation and the neural network controller.

The controller listens to a reference velocity in the topic `/cmd_vel`. You can steer and controll the robot using the rqt plugin `rqt_robot_steering`. To lauch it, run in another terminal

        ros2 run rqt_robot_steering rqt_robot_steering

## Real exeriments usage




## Troubleshooting
### bad_alloc
use a different ROS_DOMAIN_ID

        export ROS_DOMAIN_ID=42

## Tips and Tricks:

### Simplifying meshes

STL files in the mulinex_description package may be too big for most of the computers. To simplify them, you can use a dedicated script in the `mulinex_description/meshes` folder. To use it, navigate to the folder with

        cd src/mulinex_description/meshes

and run

        python3 simplify_meshes.py

This will save all the simplified meshes as `simpler_[meshname].stl` in the same folder. You can then either rename them to replace the original meshes or change the filename in the file `src/mulinex_description/urdf/links.xacro`
