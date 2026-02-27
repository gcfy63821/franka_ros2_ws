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
Launch file for trajectory subscriber joint impedance controller on real robot.

This controller subscribes to joint trajectory messages published by the
publish_pickle_trajectory.py script and follows them using joint impedance control.

Usage:
    ros2 launch franka_bringup trajectory_subscriber_joint_impedance_controller.launch.py

Or with custom config:
    ros2 launch franka_bringup trajectory_subscriber_joint_impedance_controller.launch.py \
        robot_config_file:=/path/to/config.yaml

The trajectory topic name can be set in the controllers.yaml configuration file
or via ROS parameter after launch. Default is "arm_joint_trajectory".

To publish trajectories, run:
    python3 publish_pickle_trajectory.py <pkl_file> --fps 30 --loop
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

    # Include the existing example.launch.py file with trajectory_subscriber_joint_impedance_controller
    additional_nodes.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('franka_bringup'), 'launch', 'example.launch.py'
                ])
            ),
            launch_arguments={
                'robot_config_file': robot_config_file,
                'controller_name': 'trajectory_subscriber_joint_impedance_controller',
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

