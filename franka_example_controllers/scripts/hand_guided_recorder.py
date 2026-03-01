#!/usr/bin/env python3
"""
Hand-guided robot data recorder.

This script records hand-guided robot data with waypoint marking:
1. Subscribes to RealSense camera images from ROS topic (camera must be launched separately)
2. Press 'a' at each waypoint to record (joint pose + gripper state)
3. Press 's' to finish and send recorded waypoints
4. Receives start signal from waypoint controller (when it reaches first waypoint)
5. Collects video and joint trajectory at desired FPS until stop signal

Prerequisites:
    - RealSense camera must be running via: ros2 launch realsense2_camera rs_launch.py

Usage:
    python3 hand_guided_recorder.py [--fps FPS] [--output_dir DIR] [--image_topic TOPIC]
    
Example:
    python3 hand_guided_recorder.py --fps 30 --output_dir ~/recordings
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Bool, Header
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import sys
import os
import time
import argparse
import threading
import queue
import cv2
import numpy as np
from datetime import datetime


class HandGuidedRecorder(Node):
    def __init__(self, fps=30, output_dir=None, image_topic=None):
        super().__init__('hand_guided_recorder')
        
        self.fps = fps
        self.output_dir = output_dir or os.path.expanduser("~/robot_recordings")
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_topic = image_topic or '/camera/color/image_raw'
        
        # Recording state
        self.recording_active = False
        self.waypoint_mode = True  # Start in waypoint marking mode
        self.waypoints = []  # List of waypoints (joint positions + gripper state)
        self.video_frames = []  # List of video frames
        self.joint_trajectory = []  # List of joint states during recording
        self.timestamps = []  # Timestamps for each frame
        
        # Image handling (from ROS topic)
        self.cv_bridge = CvBridge()
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.video_writer = None
        self.video_path = None
        self.image_width = 640
        self.image_height = 480
        
        # ROS subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/measured_joint_states',  # Franka robot state broadcaster topic
            self.joint_state_callback,
            10
        )
        
        # Try alternative joint state topic
        self.joint_state_sub_alt = self.create_subscription(
            JointState,
            '/NS_1/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Subscribe to RealSense camera image topic
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        self.get_logger().info(f"Subscribed to RealSense image topic: {self.image_topic}")
        
        # Subscribe to start/stop signals
        self.start_sub = self.create_subscription(
            Bool,
            '/recording_start',
            self.start_recording_callback,
            10
        )
        
        self.stop_sub = self.create_subscription(
            Bool,
            '/recording_stop',
            self.stop_recording_callback,
            10
        )
        
        # Also subscribe to finish signal as stop signal
        self.finish_sub = self.create_subscription(
            Bool,
            '/trajectory_finished',
            self.stop_recording_callback,
            10
        )
        
        # Publisher for waypoint trajectory
        self.waypoint_pub = self.create_publisher(
            JointTrajectory,
            '/arm_joint_trajectory',
            10
        )
        
        # Current joint state
        self.current_joint_state = None
        self.current_gripper_state = None
        self.joint_state_lock = threading.Lock()
        
        # Keyboard input thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        
        # Recording thread
        self.recording_thread = None
        self.recording_lock = threading.Lock()
        
        self.get_logger().info("Hand-guided recorder initialized")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info("Mode: Waypoint marking")
        self.get_logger().info("  Press 'a' to record waypoint with gripper OPEN")
        self.get_logger().info("  Press 'w' to record waypoint with gripper CLOSED")
        self.get_logger().info("  Press 's' to send waypoints")
        self.get_logger().info("Waiting for start signal from controller to begin continuous recording...")
    
    def image_callback(self, msg):
        """Callback for RealSense image messages."""
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.image_lock:
                self.latest_image = cv_image
                # Update image dimensions if needed
                if self.latest_image is not None:
                    h, w = self.latest_image.shape[:2]
                    self.image_width = w
                    self.image_height = h
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
    
    def joint_state_callback(self, msg):
        """Callback for joint state messages."""
        with self.joint_state_lock:
            # Extract arm joint positions (first 7 joints)
            if len(msg.position) >= 7:
                self.current_joint_state = list(msg.position[:7])
            else:
                self.current_joint_state = None
            
            # Extract gripper state (if available, typically last 2 joints)
            if len(msg.position) >= 9:
                # Gripper has 2 joints, take average width
                gripper_width = (msg.position[7] + msg.position[8]) * 2.0  # Convert to width
                self.current_gripper_state = {
                    'width': gripper_width,
                    'position': list(msg.position[7:9])
                }
            elif len(msg.position) == 7:
                # No gripper in message
                self.current_gripper_state = None
            else:
                self.current_gripper_state = None
    
    def start_recording_callback(self, msg):
        """Callback for start recording signal."""
        if msg.data and not self.recording_active:
            self.get_logger().info("Received start signal! Beginning continuous recording...")
            self.start_continuous_recording()
    
    def stop_recording_callback(self, msg):
        """Callback for stop recording signal."""
        if msg.data and self.recording_active:
            self.get_logger().info("Received stop signal! Stopping recording...")
            self.stop_continuous_recording()
    
    def start_continuous_recording(self):
        """Start continuous video and joint trajectory recording."""
        with self.recording_lock:
            if self.recording_active:
                return
            
            self.recording_active = True
            self.waypoint_mode = False  # Switch to continuous recording mode
            
            # Clear previous data
            self.video_frames = []
            self.joint_trajectory = []
            self.timestamps = []
            
            # Setup video writer (will use latest image dimensions)
            # Wait a bit to get first image to determine dimensions
            time.sleep(0.5)
            with self.image_lock:
                if self.latest_image is not None:
                    h, w = self.latest_image.shape[:2]
                    self.image_width = w
                    self.image_height = h
                else:
                    self.get_logger().warn("No image received yet, using default dimensions")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_path = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, self.fps, (self.image_width, self.image_height)
            )
            self.get_logger().info(f"Video will be saved to: {self.video_path} (size: {self.image_width}x{self.image_height})")
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self.recording_loop, daemon=True)
            self.recording_thread.start()
            
            self.get_logger().info(f"Continuous recording started at {self.fps} FPS")
    
    def stop_continuous_recording(self):
        """Stop continuous recording and save data."""
        with self.recording_lock:
            if not self.recording_active:
                return
            
            self.recording_active = False
            
            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)
            
            # Save video
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                self.get_logger().info(f"Video saved to: {self.video_path}")
            
            # Save joint trajectory
            if self.joint_trajectory:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                traj_path = os.path.join(self.output_dir, f"joint_trajectory_{timestamp}.npz")
                np.savez(
                    traj_path,
                    joint_positions=np.array(self.joint_trajectory),
                    timestamps=np.array(self.timestamps)
                )
                self.get_logger().info(f"Joint trajectory saved to: {traj_path}")
                self.get_logger().info(f"Recorded {len(self.joint_trajectory)} frames")
            
            self.get_logger().info("Recording stopped and data saved")
    
    def recording_loop(self):
        """Main recording loop running at specified FPS."""
        frame_time = 1.0 / self.fps
        last_time = time.time()
        
        while self.recording_active:
            current_time = time.time()
            elapsed = current_time - last_time
            
            if elapsed >= frame_time:
                # Capture frame
                frame_data = self.capture_frame()
                
                with self.recording_lock:
                    if self.recording_active:
                        # Record joint state
                        with self.joint_state_lock:
                            if self.current_joint_state:
                                self.joint_trajectory.append(self.current_joint_state.copy())
                                self.timestamps.append(current_time)
                        
                        # Record video frame
                        if frame_data is not None and self.video_writer:
                            self.video_writer.write(frame_data)
                            self.video_frames.append(frame_data.copy())
                
                last_time = current_time
            else:
                # Sleep to maintain FPS
                time.sleep(frame_time - elapsed)
    
    def capture_frame(self):
        """Get latest frame from RealSense camera (via ROS topic)."""
        with self.image_lock:
            if self.latest_image is not None:
                return self.latest_image.copy()
        return None
    
    def keyboard_listener(self):
        """Listen for keyboard input in waypoint marking mode."""
        try:
            import select
            import tty
            import termios
            
            # Set terminal to raw mode
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())
                
                self.get_logger().info("Keyboard listener started:")
                self.get_logger().info("  'a' - record waypoint with gripper OPEN")
                self.get_logger().info("  'w' - record waypoint with gripper CLOSED")
                self.get_logger().info("  's' - send waypoints")
                self.get_logger().info("  'q' - quit")
                
                while rclpy.ok():
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        
                        if key == 'a' or key == 'A':
                            # Record waypoint with gripper open
                            self.record_waypoint(gripper_open=True)
                        elif key == 'w' or key == 'W':
                            # Record waypoint with gripper closed
                            self.record_waypoint(gripper_open=False)
                        elif key == 's' or key == 'S':
                            # Send waypoints
                            self.send_waypoints()
                        elif key == 'q' or key == 'Q':
                            # Quit
                            self.get_logger().info("Quitting...")
                            break
                        elif key == '\x03':  # Ctrl+C
                            break
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except ImportError:
            # Fallback for systems without termios (e.g., Windows)
            self.get_logger().warn("termios not available. Keyboard input disabled. Use ROS topics for control.")
        except Exception as e:
            self.get_logger().error(f"Error in keyboard listener: {e}")
    
    def record_waypoint(self, gripper_open=True):
        """Record current position as a waypoint with specified gripper state.
        
        Args:
            gripper_open: True for gripper open, False for gripper closed
        """
        with self.joint_state_lock:
            if self.current_joint_state is None:
                self.get_logger().warn("No joint state available to record")
                return
            
            waypoint = {
                'joint_positions': self.current_joint_state.copy(),
                'gripper_open': gripper_open,
                'gripper_state': self.current_gripper_state.copy() if self.current_gripper_state else None,
                'timestamp': time.time()
            }
            
            self.waypoints.append(waypoint)
            gripper_status = "OPEN" if gripper_open else "CLOSED"
            self.get_logger().info(
                f"Recorded waypoint {len(self.waypoints)} (gripper {gripper_status}): "
                f"joints={[f'{x:.3f}' for x in self.current_joint_state]}"
            )
            if self.current_gripper_state:
                self.get_logger().info(f"  current_gripper_width={self.current_gripper_state['width']:.4f} m")
    
    def send_waypoints(self):
        """Send recorded waypoints as a trajectory message."""
        if not self.waypoints:
            self.get_logger().warn("No waypoints recorded to send")
            return
        
        trajectory = JointTrajectory()
        trajectory.header = Header()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.header.frame_id = "base"
        
        # Set joint names (assuming Franka arm + gripper state as 8th "joint")
        # Gripper state: 0.0 = closed, 1.0 = open
        trajectory.joint_names = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
            "fr3_joint5", "fr3_joint6", "fr3_joint7", "gripper_state"
        ]
        
        # Add waypoints
        for waypoint in self.waypoints:
            point = JointTrajectoryPoint()
            # Include 7 joint positions + gripper state (0.0 = closed, 1.0 = open)
            positions = waypoint['joint_positions'].copy()
            gripper_value = 1.0 if waypoint['gripper_open'] else 0.0
            positions.append(gripper_value)
            point.positions = positions
            point.velocities = [0.0] * 8
            point.accelerations = [0.0] * 8
            point.effort = [0.0] * 8
            point.time_from_start = Duration(sec=0, nanosec=0)
            trajectory.points.append(point)
        
        # Log gripper states
        gripper_states = ["OPEN" if wp['gripper_open'] else "CLOSED" for wp in self.waypoints]
        self.get_logger().info(f"Waypoint gripper states: {gripper_states}")
        
        self.waypoint_pub.publish(trajectory)
        self.get_logger().info(f"Sent {len(self.waypoints)} waypoints as trajectory")
        
        # Optionally clear waypoints after sending
        # self.waypoints = []
    
    def cleanup(self):
        """Cleanup resources."""
        self.recording_active = False
        
        if self.video_writer:
            self.video_writer.release()
        
        self.get_logger().info("Recorder cleaned up")


def main(args=None):
    parser = argparse.ArgumentParser(description='Hand-guided robot data recorder')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Recording frame rate (default: 30)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for recordings (default: ~/robot_recordings)')
    parser.add_argument('--image_topic', type=str, default='/camera/color/image_raw',
                       help='ROS topic for camera images (default: /camera/color/image_raw)')
    
    args = parser.parse_args()
    
    rclpy.init(args=sys.argv)
    
    recorder = HandGuidedRecorder(fps=args.fps, output_dir=args.output_dir, image_topic=args.image_topic)
    
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.cleanup()
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

