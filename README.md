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

        

This is a project built on devcontainer. To install it, you need to have docker and vscode installed with the extension `ms-vscode-remote.remote-containers`.
 Then, you can open the project in vscode and click on the button on the bottom left corner. This will build the devcontainer and install all the dependencies.


> **WARNING:** The project is configured to run with GPU access. If you don't have an Nvidia GPU, you need to change the devcontainer.json and remove the `--gpus=all` flag from the docker run command.
By doing this you will not also be able to run the neural network controller.

## Usage
**Run all the commands from the terminal inside vscode**

First, you need to build the project. To do so, you can use the command 

        colcon build --symlink-install

This will build the packages and create a symlink to the install folder.

To run the simulation, you can use 

        ros2 launch mulinex_gazebo launch sim.launch.py
        
This will launch the simulation and the PD controller.
To run the neural network controller, you can use `ros2 launch rlg_quad_controller mulinex_simulation.launch.py`. This will launch the simulation and the neural network controller.

To perform any

## TODO:
- [ ] add a script to simplify meshes