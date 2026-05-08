#  Copyright (c) 2026 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

############################################################################
# Convenience launcher that brings up the TeachReplayController, which has
# two modes (TEACH = zero torques, REPLAY = dense joint trajectory tracking).
#
# Mode is switched at runtime by publishing on /teach_replay/mode:
#   ros2 topic pub --once /teach_replay/mode std_msgs/String "{data: teach}"
#   ros2 topic pub --once /teach_replay/mode std_msgs/String "{data: replay}"
#
# Trajectories are sent on /teach_replay/trajectory as
# trajectory_msgs/JointTrajectory with time_from_start populated for each
# point. The controller publishes /teach_replay/replay_started and
# /teach_replay/replay_finished as edge-triggered Bool signals.
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
                'controller_name': 'teach_replay_controller',
                'robot_config_file': LaunchConfiguration('robot_config_file'),
            }.items(),
        ),
    ])
