import imp
import os
from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,ExecuteProcess,TimerAction
from launch.substitutions import Command,LaunchConfiguration,PathJoinSubstitution
from launch.conditions import IfCondition,UnlessCondition

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    load_jnt_controll = ExecuteProcess(
        cmd=[["ros2 control load_controller joint_controller --set-state active"]],
        shell=True,
        output = "screen"
    )
    load_stt_controll = ExecuteProcess(
        cmd=[["ros2 control load_controller state_broadcaster --set-state active"]],
        shell=True,
        output = "screen"
    )

    delayed_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="sim_to_real",
                executable="nn_traj_generator",
            )
        ]
    )

    return LaunchDescription(
        [
            load_jnt_controll,
            load_stt_controll,
            delayed_node
        ]
    )