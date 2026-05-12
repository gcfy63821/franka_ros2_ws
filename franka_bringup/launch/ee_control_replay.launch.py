#  Copyright (c) 2026 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

############################################################################
# Convenience launcher for FK replay through Cartesian pose control.
#
# The spawned controller is ee_pose_replay_controller. Trajectories are sent on
# /ee_pose_replay/trajectory as trajectory_msgs/MultiDOFJointTrajectory, and
# mode is switched by publishing "hold" or "replay" on /ee_pose_replay/mode.
############################################################################

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('franka_bringup'), 'config', 'franka.config.yaml'
            ]),
            description='Path to the robot configuration file',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('franka_bringup'), 'launch', 'example.launch.py'
                ])
            ),
            launch_arguments={
                'controller_name': 'ee_pose_replay_controller',
                'robot_config_file': LaunchConfiguration('robot_config_file'),
            }.items(),
        ),
    ])
