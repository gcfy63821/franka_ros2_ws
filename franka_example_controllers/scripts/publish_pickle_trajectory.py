#!/usr/bin/env python3
"""
Publish trajectory data from pickle file frame by frame.

This script loads trajectory data from a pickle file and publishes:
- Hand joint positions (22 joints)
- Arm joint positions (7 joints) - as JointState
- Arm joint trajectory (7 joints) - as JointTrajectory for controllers
- Wrist translation (x, y, z)
- Wrist orientation (quaternion x, y, z, w)

The trajectory is published frame by frame at a configurable FPS in a repeating loop.

Usage:
    python3 publish_pickle_trajectory.py <pkl_file> [--fps FPS] [--loop]
    
Example:
    python3 publish_pickle_trajectory.py trajectory_ep0002_20260115_151217.pkl --fps 30 --loop
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration
from tf2_ros import TransformBroadcaster
import sys
import os
import time
import argparse
import numpy as np
try:
    import joblib
except ImportError:
    import pickle as joblib


class PickleTrajectoryPublisher(Node):
    # Hand joint names corresponding to the saved sequence in pickle file
    HAND_JOINT_NAMES = [
        "left_thumb_CMC_FE",
        "left_thumb_CMC_AA",
        "left_thumb_MCP_FE",
        "left_thumb_MCP_AA",
        "left_thumb_IP",
        "left_index_MCP_FE",
        "left_index_MCP_AA",
        "left_index_PIP",
        "left_index_DIP",
        "left_middle_MCP_FE",
        "left_middle_MCP_AA",
        "left_middle_PIP",
        "left_middle_DIP",
        "left_ring_MCP_FE",
        "left_ring_MCP_AA",
        "left_ring_PIP",
        "left_ring_DIP",
        "left_pinky_CMC",
        "left_pinky_MCP_FE",
        "left_pinky_MCP_AA",
        "left_pinky_PIP",
        "left_pinky_DIP",
    ]
    
    def __init__(self, pkl_file: str, fps: float = 30.0, loop: bool = True):
        super().__init__('pickle_trajectory_publisher')
        
        self.pkl_file = pkl_file
        self.fps = fps
        self.loop = loop
        self.frame_duration = 1.0 / fps
        
        # Load trajectory data
        self.data = self.load_trajectory_data(pkl_file)
        self.trajectory_length = self.data['trajectory_length']
        self.current_frame = 0
        
        # Create publishers
        self.hand_joint_pub = self.create_publisher(
            JointState,
            'hand_joint_positions',
            10
        )
        
        self.arm_joint_pub = self.create_publisher(
            JointState,
            'arm_joint_positions',
            10
        )
        
        # Publisher for joint trajectory (for controllers)
        # Use absolute topic name to work across namespaces
        self.arm_joint_trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/arm_joint_trajectory',  # Absolute topic name (starts with /)
            10
        )
        
        self.wrist_pose_pub = self.create_publisher(
            PoseStamped,
            'wrist_pose',
            10
        )
        
        # Create TF broadcaster for wrist pose
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Frame names
        self.base_frame = "fr3_link0"  # Base frame
        self.wrist_frame = "wrist_pose"  # Wrist/end effector frame
        
        self.get_logger().info(f'Pickle trajectory publisher started')
        self.get_logger().info(f'  File: {pkl_file}')
        self.get_logger().info(f'  Trajectory length: {self.trajectory_length} frames')
        self.get_logger().info(f'  FPS: {fps} ({self.frame_duration*1000:.2f} ms per frame)')
        self.get_logger().info(f'  Loop: {loop}')
        self.get_logger().info(f'  Topics:')
        self.get_logger().info(f'    - hand_joint_positions (sensor_msgs/JointState)')
        self.get_logger().info(f'    - arm_joint_positions (sensor_msgs/JointState)')
        self.get_logger().info(f'    - arm_joint_trajectory (trajectory_msgs/JointTrajectory)')
        self.get_logger().info(f'    - wrist_pose (geometry_msgs/PoseStamped)')
        self.get_logger().info(f'  TF:')
        self.get_logger().info(f'    - {self.base_frame} -> {self.wrist_frame}')
        
        # Create timer to publish frames
        self.timer = self.create_timer(self.frame_duration, self.publish_frame)
        
        # Wait a bit for subscribers to connect
        time.sleep(1.0)
        
    def load_trajectory_data(self, pkl_path: str):
        """Load trajectory data from pickle file."""
        self.get_logger().info(f'Loading trajectory from: {pkl_path}')
        try:
            data = joblib.load(pkl_path)
            self.get_logger().info(f'Loaded data keys: {list(data.keys())}')
            self.get_logger().info(f'Trajectory length: {data["trajectory_length"]}')
            
            # Verify expected keys
            required_keys = ['trajectory_length', 'hand_joint_positions', 'arm_joint_positions', 
                           'wrist_translations', 'wrist_orientations']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key: {key}")
            
            # Convert to numpy arrays if needed
            if not isinstance(data['hand_joint_positions'], np.ndarray):
                data['hand_joint_positions'] = np.array(data['hand_joint_positions'])
            if not isinstance(data['arm_joint_positions'], np.ndarray):
                data['arm_joint_positions'] = np.array(data['arm_joint_positions'])
            if not isinstance(data['wrist_translations'], np.ndarray):
                data['wrist_translations'] = np.array(data['wrist_translations'])
            if not isinstance(data['wrist_orientations'], np.ndarray):
                data['wrist_orientations'] = np.array(data['wrist_orientations'])
            
            # Verify shapes
            expected_hand_joints = data['hand_joint_positions'].shape[1] if len(data['hand_joint_positions'].shape) > 1 else 0
            expected_arm_joints = data['arm_joint_positions'].shape[1] if len(data['arm_joint_positions'].shape) > 1 else 0
            
            # Verify hand joint count matches expected names
            if expected_hand_joints != len(self.HAND_JOINT_NAMES):
                self.get_logger().warn(
                    f'Hand joint count mismatch: data has {expected_hand_joints} joints, '
                    f'but {len(self.HAND_JOINT_NAMES)} joint names provided. '
                    f'Using first {min(expected_hand_joints, len(self.HAND_JOINT_NAMES))} names.'
                )
            
            self.get_logger().info(f'  Hand joints: {expected_hand_joints}')
            self.get_logger().info(f'  Arm joints: {expected_arm_joints}')
            
            return data
        except Exception as e:
            self.get_logger().error(f'Error loading trajectory: {str(e)}')
            raise
    
    def publish_frame(self):
        """Publish current frame data."""
        if self.current_frame >= self.trajectory_length:
            if self.loop:
                self.current_frame = 0
                self.get_logger().info('Looping trajectory...')
            else:
                self.get_logger().info('Trajectory complete. Stopping.')
                self.timer.cancel()
                return
        
        # Get current frame data
        hand_joints = self.data['hand_joint_positions'][self.current_frame]
        arm_joints = self.data['arm_joint_positions'][self.current_frame]
        wrist_translation = self.data['wrist_translations'][self.current_frame]
        wrist_orientation = self.data['wrist_orientations'][self.current_frame]
        
        # Get current time
        now = self.get_clock().now()
        
        # Publish hand joint positions
        hand_msg = JointState()
        hand_msg.header = Header()
        hand_msg.header.stamp = now.to_msg()
        hand_msg.header.frame_id = "hand"
        # Use actual joint names, truncate or pad if needed
        num_joints = len(hand_joints)
        if num_joints <= len(self.HAND_JOINT_NAMES):
            hand_msg.name = self.HAND_JOINT_NAMES[:num_joints]
        else:
            # If more joints than names, use names + generic for extras
            hand_msg.name = self.HAND_JOINT_NAMES + [f'hand_joint_{i}' for i in range(len(self.HAND_JOINT_NAMES), num_joints)]
        hand_msg.position = hand_joints.tolist()
        hand_msg.velocity = []  # Not available in data
        hand_msg.effort = []    # Not available in data
        self.hand_joint_pub.publish(hand_msg)
        
        # Publish arm joint positions
        arm_msg = JointState()
        arm_msg.header = Header()
        arm_msg.header.stamp = now.to_msg()
        arm_msg.header.frame_id = "arm"
        # Use fr3_joint names to match controller configuration
        arm_msg.name = [
            'fr3_joint1',
            'fr3_joint2',
            'fr3_joint3',
            'fr3_joint4',
            'fr3_joint5',
            'fr3_joint6',
            'fr3_joint7'
        ]
        arm_msg.position = arm_joints.tolist()
        arm_msg.velocity = []  # Not available in data
        arm_msg.effort = []    # Not available in data
        self.arm_joint_pub.publish(arm_msg)
        
        # Publish arm joint trajectory (for controllers) - frame by frame
        # Create JointTrajectory message with current frame's joint positions
        joint_trajectory_msg = JointTrajectory()
        joint_trajectory_msg.header = Header()
        joint_trajectory_msg.header.stamp = now.to_msg()
        joint_trajectory_msg.header.frame_id = "fr3_link0"
        
        # Set joint names (Franka arm has 7 joints)
        joint_trajectory_msg.joint_names = [
            'fr3_joint1',
            'fr3_joint2',
            'fr3_joint3',
            'fr3_joint4',
            'fr3_joint5',
            'fr3_joint6',
            'fr3_joint7'
        ]
        
        # Create trajectory point with current frame's joint positions
        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = arm_joints.tolist()
        
        # Calculate time from start based on frame index and FPS
        # Each frame represents 1/fps seconds
        frame_time_sec = self.current_frame / self.fps
        frame_time_nsec = int((frame_time_sec - int(frame_time_sec)) * 1e9)
        trajectory_point.time_from_start = Duration(
            sec=int(frame_time_sec),
            nanosec=frame_time_nsec
        )
        
        # Publish single point trajectory for current frame
        # Controllers can receive this frame-by-frame trajectory
        joint_trajectory_msg.points = [trajectory_point]
        self.arm_joint_trajectory_pub.publish(joint_trajectory_msg)
        
        # Publish wrist pose (translation + orientation)
        wrist_msg = PoseStamped()
        wrist_msg.header = Header()
        wrist_msg.header.stamp = now.to_msg()
        wrist_msg.header.frame_id = self.base_frame
        
        # Set translation
        wrist_msg.pose.position = Point()
        wrist_msg.pose.position.x = float(wrist_translation[0])
        wrist_msg.pose.position.y = float(wrist_translation[1])
        wrist_msg.pose.position.z = float(wrist_translation[2])
        
        # Set orientation (quaternion: x, y, z, w in data, but ROS uses w, x, y, z)
        wrist_msg.pose.orientation = Quaternion()
        wrist_msg.pose.orientation.w = float(wrist_orientation[3])  # w component
        wrist_msg.pose.orientation.x = float(wrist_orientation[0])  # x component
        wrist_msg.pose.orientation.y = float(wrist_orientation[1])  # y component
        wrist_msg.pose.orientation.z = float(wrist_orientation[2])  # z component
        
        self.wrist_pose_pub.publish(wrist_msg)
        
        # Publish TF transform for wrist pose
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = self.base_frame
        t.child_frame_id = self.wrist_frame
        
        # Set translation
        t.transform.translation.x = float(wrist_translation[0])
        t.transform.translation.y = float(wrist_translation[1])
        t.transform.translation.z = float(wrist_translation[2])
        
        # Set orientation (quaternion: x, y, z, w in data, but ROS uses w, x, y, z)
        t.transform.rotation.w = float(wrist_orientation[3])  # w component
        t.transform.rotation.x = float(wrist_orientation[0])  # x component
        t.transform.rotation.y = float(wrist_orientation[1])  # y component
        t.transform.rotation.z = float(wrist_orientation[2])  # z component
        
        self.tf_broadcaster.sendTransform(t)
        
        # Log progress every 10 frames
        if self.current_frame % 10 == 0:
            self.get_logger().info(f'Published frame {self.current_frame}/{self.trajectory_length-1}')
        
        self.current_frame += 1


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Publish trajectory data from pickle file frame by frame',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Publish at 30 FPS with looping
  python3 publish_pickle_trajectory.py trajectory.pkl --fps 30 --loop
  
  # Publish at 60 FPS without looping
  python3 publish_pickle_trajectory.py trajectory.pkl --fps 60 --no-loop
        """
    )
    
    parser.add_argument('pkl_file', type=str, default="/home/robot/workspace/franka_control/src/trajectory_ep0002_20260115_151217.pkl", 
                        help='Path to pickle trajectory file')
    parser.add_argument('--fps', type=float, default=30.0, 
                       help='Publishing frequency in frames per second (default: 30.0)')
    parser.add_argument('--loop', action='store_true', default=True,
                       help='Loop the trajectory continuously (default: True)')
    parser.add_argument('--no-loop', dest='loop', action='store_false',
                       help='Do not loop the trajectory')
    
    args = parser.parse_args(args)
    
    # Validate file exists
    if not os.path.exists(args.pkl_file):
        print(f"Error: File not found: {args.pkl_file}")
        sys.exit(1)
    
    # Validate FPS
    if args.fps <= 0:
        print(f"Error: FPS must be positive, got {args.fps}")
        sys.exit(1)
    
    # Initialize ROS2
    rclpy.init(args=None)
    
    try:
        publisher = PickleTrajectoryPublisher(
            pkl_file=args.pkl_file,
            fps=args.fps,
            loop=args.loop
        )
        
        # Spin the node
        rclpy.spin(publisher)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

