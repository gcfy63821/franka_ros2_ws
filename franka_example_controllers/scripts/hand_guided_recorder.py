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
from rclpy.qos import qos_profile_sensor_data
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
import cv2
import numpy as np
from datetime import datetime


class HandGuidedRecorder(Node):
    def __init__(self, fps=30, output_dir=None, image_topic=None):
        super().__init__('hand_guided_recorder')
        
        self.fps = fps
        self.output_dir = output_dir or os.path.expanduser("~/robot_recordings")
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_topic = image_topic or '/camera/camera/color/image_raw'
        
        # Recording state
        self.recording_active = False
        self.waypoint_mode = True  # Start in waypoint marking mode
        self.waypoints = []  # List of waypoints (joint positions + gripper state)
        self.joint_trajectory = []  # List of joint states during recording
        self.timestamps = []  # Timestamps for each frame
        self.frame_count = 0
        self.session_dir = None  # Per-session output folder
        
        # Image handling (from ROS topic)
        self.cv_bridge = CvBridge()
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.image_recv_count = 0
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
        
        # Subscribe to RealSense camera image topic (BEST_EFFORT to match RealSense)
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
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

        # OpenCV preview window
        cv2.namedWindow('Camera Preview', cv2.WINDOW_AUTOSIZE)
        self.display_timer = self.create_timer(1.0 / 30.0, self.display_callback)

        self.get_logger().info("Hand-guided recorder initialized")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info("Mode: Waypoint marking")
        self.get_logger().info("  'a' - record waypoint with gripper OPEN")
        self.get_logger().info("  'w' - record waypoint with gripper CLOSED")
        self.get_logger().info("  's' - send waypoints")
        self.get_logger().info("  'd' - backup waypoints & start new trajectory")
        self.get_logger().info("Waiting for start signal from controller to begin continuous recording...")
    
    def image_callback(self, msg):
        """Callback for RealSense image messages."""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            with self.image_lock:
                self.latest_image = cv_image
                h, w = cv_image.shape[:2]
                self.image_width = w
                self.image_height = h

            self.image_recv_count += 1
            if self.image_recv_count == 1:
                self.get_logger().info(
                    f"First image received! Size: {w}x{h}, encoding: {msg.encoding}"
                )
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
    
    def display_callback(self):
        """Timer callback to refresh the preview window."""
        with self.image_lock:
            frame = self.latest_image.copy() if self.latest_image is not None else None

        if frame is None:
            # Show placeholder when no image
            frame = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for camera...", (50, self.image_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Draw status overlay
        # Recording indicator
        if self.recording_active:
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(frame, f"REC  frames: {self.frame_count}", (50, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            status = "WAYPOINT MODE" if self.waypoint_mode else "IDLE"
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Waypoint count
        cv2.putText(frame, f"Waypoints: {len(self.waypoints)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Camera Preview', frame)
        cv2.waitKey(1)

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
    
    def _create_session_dir(self):
        """Create a timestamped session folder. All files for this session go here."""
        if self.session_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(self.output_dir, f"session_{timestamp}")
            os.makedirs(self.session_dir, exist_ok=True)
            self.get_logger().info(f"Session folder: {self.session_dir}")
        return self.session_dir

    def reset_waypoints(self):
        """Backup current waypoints to file and start a new trajectory."""
        if self.waypoints:
            # Save old waypoints as backup
            self._create_session_dir()
            backup_path = os.path.join(self.session_dir, "waypoints_backup.npz")
            joint_positions = [wp['joint_positions'] for wp in self.waypoints]
            gripper_open = [wp['gripper_open'] for wp in self.waypoints]
            wp_timestamps = [wp['timestamp'] for wp in self.waypoints]
            np.savez(
                backup_path,
                joint_positions=np.array(joint_positions),
                gripper_open=np.array(gripper_open),
                timestamps=np.array(wp_timestamps),
            )
            self.get_logger().info(
                f"Backed up {len(self.waypoints)} waypoints to: {backup_path}"
            )

        # Reset for new trajectory
        self.waypoints = []
        self.session_dir = None  # Next session gets a new folder
        self.get_logger().info("Waypoints cleared. Ready to record new trajectory.")

    def start_continuous_recording(self):
        """Start continuous video and joint trajectory recording."""
        if self.recording_active:
            return

        self.waypoint_mode = False  # Switch to continuous recording mode

        # Clear previous data
        self.joint_trajectory = []
        self.timestamps = []
        self.frame_count = 0

        # Ensure session folder exists
        session = self._create_session_dir()

        # Get image dimensions from latest frame
        with self.image_lock:
            if self.latest_image is not None:
                h, w = self.latest_image.shape[:2]
                self.image_width = w
                self.image_height = h
            else:
                self.get_logger().warn("No image received yet, using default 640x480")

        self.video_path = os.path.join(session, "recording.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            self.video_path, fourcc, self.fps, (self.image_width, self.image_height)
        )

        if not self.video_writer.isOpened():
            self.get_logger().error(f"Failed to open VideoWriter: {self.video_path}")
            return

        self.get_logger().info(f"Video: {self.video_path} ({self.image_width}x{self.image_height})")

        # Set flag and start thread AFTER writer is confirmed open
        self.recording_active = True
        self.recording_thread = threading.Thread(target=self.recording_loop, daemon=True)
        self.recording_thread.start()

        self.get_logger().info(f"Continuous recording started at {self.fps} FPS")
    
    def stop_continuous_recording(self):
        """Stop continuous recording and save data."""
        if not self.recording_active:
            return

        # Signal thread to stop first, then join (no lock held → no deadlock)
        self.recording_active = False

        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=5.0)

        # Thread has exited, safe to release writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.get_logger().info(f"Video saved: {self.video_path} ({self.frame_count} frames)")

        # Save joint trajectory to session folder
        if self.joint_trajectory and self.session_dir:
            traj_path = os.path.join(self.session_dir, "joint_trajectory.npz")
            np.savez(
                traj_path,
                joint_positions=np.array(self.joint_trajectory),
                timestamps=np.array(self.timestamps)
            )
            self.get_logger().info(f"Joint trajectory saved: {traj_path} ({len(self.joint_trajectory)} frames)")

        self.get_logger().info(f"Recording stopped. All data saved to: {self.session_dir}")
    
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

                # Record joint state
                with self.joint_state_lock:
                    if self.current_joint_state:
                        self.joint_trajectory.append(self.current_joint_state.copy())
                        self.timestamps.append(current_time)

                # Write video frame directly (no extra copy in memory)
                if frame_data is not None and self.video_writer:
                    self.video_writer.write(frame_data)
                    self.frame_count += 1

                last_time = current_time
            else:
                time.sleep(max(0.0, frame_time - elapsed))
    
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
                self.get_logger().info("  'd' - backup waypoints & start new trajectory")
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
                        elif key == 'd' or key == 'D':
                            # Backup and reset waypoints
                            self.reset_waypoints()
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

        # Create session folder for this trajectory
        session = self._create_session_dir()

        # Save waypoints to session folder
        joint_positions = [wp['joint_positions'] for wp in self.waypoints]
        gripper_open = [wp['gripper_open'] for wp in self.waypoints]
        wp_timestamps = [wp['timestamp'] for wp in self.waypoints]
        wp_path = os.path.join(session, "waypoints.npz")
        np.savez(
            wp_path,
            joint_positions=np.array(joint_positions),
            gripper_open=np.array(gripper_open),
            timestamps=np.array(wp_timestamps),
        )
        self.get_logger().info(f"Waypoints saved: {wp_path}")

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
    
    def cleanup(self):
        """Cleanup resources."""
        self.recording_active = False

        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()
        self.get_logger().info("Recorder cleaned up")


def main(args=None):
    parser = argparse.ArgumentParser(description='Hand-guided robot data recorder')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Recording frame rate (default: 30)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for recordings (default: ~/robot_recordings)')
    parser.add_argument('--image_topic', type=str, default='/camera/camera/color/image_raw',
                       help='ROS topic for camera images (default: /camera/camera/color/image_raw)')
    
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
