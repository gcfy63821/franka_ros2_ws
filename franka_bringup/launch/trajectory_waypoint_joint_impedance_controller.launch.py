#  Copyright (c) 2025 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Launch file for trajectory waypoint joint impedance controller on real robot.

This controller receives a full joint trajectory from a ROS topic, moves to each
waypoint sequentially using minimum jerk trajectories, and controls the gripper
based on the recorded gripper state in the trajectory.

Usage:
    ros2 launch franka_bringup trajectory_waypoint_joint_impedance_controller.launch.py

Or with custom config:
    ros2 launch franka_bringup trajectory_waypoint_joint_impedance_controller.launch.py \
        robot_config_file:=/path/to/config.yaml

The controller subscribes to /arm_joint_trajectory topic and expects trajectory messages
with 8 positions (7 joints + gripper state where 0.0 = closed, 1.0 = open).

The controller publishes:
    - /recording_start: When reaching the first waypoint
    - /trajectory_finished: When all waypoints are completed
"""

import sys
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

# Add the path to the `utils` folder
package_share = get_package_share_directory('franka_bringup')
utils_path = os.path.join(package_share, '..', '..', 'lib', 'franka_bringup', 'utils')
sys.path.append(os.path.abspath(utils_path))

from launch_utils import load_yaml  # noqa: E402


def generate_robot_nodes(context):
    additional_nodes = []
    # Get the arguments from the launch configuration
    robot_config_file = LaunchConfiguration('robot_config_file').perform(context)

    # Include the existing example.launch.py file with trajectory_waypoint_joint_impedance_controller
    additional_nodes.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('franka_bringup'), 'launch', 'example.launch.py'
                ])
            ),
            launch_arguments={
                'robot_config_file': robot_config_file,
                'controller_name': 'trajectory_waypoint_joint_impedance_controller',
            }.items(),
        )
    )

    return additional_nodes


def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'robot_config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('franka_bringup'), 'config', 'franka.config.yaml'
            ]),
            description='Path to the robot configuration file to load',
        ),
        # Generate robot nodes
        OpaqueFunction(function=generate_robot_nodes),
    ])

