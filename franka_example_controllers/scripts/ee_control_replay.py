#!/usr/bin/python3
"""Replay FK end-effector pose files through EePoseReplayController.

Input layout:
    <data_dir>/traj_N/replay/EE_pose_FK.npz
    <data_dir>/traj_N/replay/gripper_events.npz   (preferred, optional)
    <data_dir>/traj_N/teach/gripper_events.npz    (fallback, optional)

Output layout:
    <data_dir>/traj_N/FK_state/tcp_pose.npz
    <data_dir>/traj_N/FK_state/joint_velocity.npz  (N, 9): 7 arm joints + 2 fingers
    <data_dir>/traj_N/FK_state/joint_pos.npz       (N, 9): 7 arm joints + 2 fingers
"""

import argparse
import os
import re
import sys
import threading
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Transform, Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Header, String
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

try:
    from franka_msgs.action import Grasp, Homing, Move
    from franka_msgs.msg import FrankaRobotState
except ImportError as exc:
    print(f"franka_msgs is required; source the workspace first: {exc}", file=sys.stderr)
    raise


ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
TRAJ_DIR_PATTERN = re.compile(r"^traj_(\d+)$")
DEFAULT_POSE_KEYS = (
    "end_effector_poses",
    "ee_pose",
    "ee_poses",
    "tcp_pose",
    "tcp_poses",
    "poses",
)
POSE_FILENAMES = ("EE_pose_FK.npz", "ee_pose_FK.npz")


def duration_from_seconds(seconds):
    seconds = max(0.0, float(seconds))
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return DurationMsg(sec=sec, nanosec=nanosec)


def stringify_action(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class EeControlReplay(Node):
    STATE_IDLE = "IDLE"
    STATE_WAITING_FOR_START = "WAITING_FOR_START"
    STATE_REPLAYING = "REPLAYING"

    def __init__(self, args):
        super().__init__("ee_control_replay")
        self.args = args
        self.record_rate = float(args.record_rate)
        self.input_rate = float(args.input_rate)
        self.trajectory_smoothing_window = int(args.trajectory_smoothing_window)

        self.lock = threading.Lock()
        self.latest_tcp_pose = None
        self.latest_arm_q = None
        self.latest_arm_dq = None
        self.latest_gripper_q = None
        self.latest_gripper_dq = None
        self.previous_gripper_q = None
        self.previous_gripper_stamp = None

        self.state = self.STATE_IDLE
        self.current_traj_dir = None
        self.current_events = []
        self.replay_t0 = None
        self.recording = False
        self.started_event = threading.Event()
        self.finished_event = threading.Event()
        self.scheduled_event_timers = []

        self.tcp_times = []
        self.tcp_poses = []
        self.q_times = []
        self.q_values = []
        self.dq_times = []
        self.dq_values = []
        self.gripper_event_log = []

        qos = 10
        self.mode_pub = self.create_publisher(String, args.mode_topic, qos)
        self.traj_pub = self.create_publisher(
            MultiDOFJointTrajectory, args.trajectory_topic, qos
        )
        self.started_sub = self.create_subscription(
            Bool, args.start_topic, self.replay_started_cb, qos
        )
        self.finished_sub = self.create_subscription(
            Bool, args.finish_topic, self.replay_finished_cb, qos
        )
        self.robot_state_sub = self.create_subscription(
            FrankaRobotState,
            args.robot_state_topic,
            self.robot_state_cb,
            qos,
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            args.joint_state_topic,
            self.joint_state_cb,
            qos,
        )
        self.record_timer = self.create_timer(1.0 / self.record_rate, self.record_tick)

        self.move_client = ActionClient(self, Move, args.gripper_move_action)
        self.grasp_client = ActionClient(self, Grasp, args.gripper_grasp_action)
        self.homing_client = ActionClient(self, Homing, args.gripper_homing_action)

        self.get_logger().info(
            f"EE control replay ready. trajectory_topic={args.trajectory_topic}"
        )

    # ------------------------------------------------------------------
    # State subscriptions
    # ------------------------------------------------------------------
    def _extract_arm_joint_vector(self, msg: JointState, field_name):
        values = getattr(msg, field_name)
        if len(values) == 0:
            return None

        arm = [None] * 7
        if msg.name:
            for i, name in enumerate(msg.name):
                if i >= len(values):
                    continue
                try:
                    joint_idx = ARM_JOINT_NAMES.index(name)
                except ValueError:
                    continue
                arm[joint_idx] = float(values[i])
        elif len(values) >= 7:
            arm = [float(x) for x in values[:7]]

        if all(v is not None for v in arm):
            return arm
        return None

    def _extract_gripper_joint_vector(self, msg: JointState, field_name):
        values = getattr(msg, field_name)
        if len(values) == 0:
            return None

        gripper = [None, None]
        if msg.name:
            for i, name in enumerate(msg.name):
                if i >= len(values):
                    continue
                if FINGER_JOINT_NAMES[0] in name:
                    gripper[0] = float(values[i])
                elif FINGER_JOINT_NAMES[1] in name:
                    gripper[1] = float(values[i])
        elif len(values) >= 9:
            gripper = [float(values[7]), float(values[8])]

        if all(v is not None for v in gripper):
            return gripper
        return None

    def _estimate_gripper_velocity(self, msg: JointState, gripper_q):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = time.time()
        if self.previous_gripper_q is None or self.previous_gripper_stamp is None:
            self.previous_gripper_q = list(gripper_q)
            self.previous_gripper_stamp = stamp
            return [0.0, 0.0]

        dt = stamp - self.previous_gripper_stamp
        if dt <= 1e-6:
            return [0.0, 0.0]
        dq = [
            (gripper_q[0] - self.previous_gripper_q[0]) / dt,
            (gripper_q[1] - self.previous_gripper_q[1]) / dt,
        ]
        self.previous_gripper_q = list(gripper_q)
        self.previous_gripper_stamp = stamp
        return dq

    @staticmethod
    def _pose7_from_pose_stamped(pose_stamped):
        pose = pose_stamped.pose
        return [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ]

    def robot_state_cb(self, msg: FrankaRobotState):
        tcp_pose = self._pose7_from_pose_stamped(msg.o_t_ee)
        arm = self._extract_arm_joint_vector(msg.measured_joint_state, "position")
        arm_dq = self._extract_arm_joint_vector(msg.measured_joint_state, "velocity")
        with self.lock:
            self.latest_tcp_pose = tcp_pose
            if arm is not None:
                self.latest_arm_q = arm
            if arm_dq is not None:
                self.latest_arm_dq = arm_dq

    def joint_state_cb(self, msg: JointState):
        arm = self._extract_arm_joint_vector(msg, "position")
        arm_dq = self._extract_arm_joint_vector(msg, "velocity")
        gripper = self._extract_gripper_joint_vector(msg, "position")
        gripper_dq = self._extract_gripper_joint_vector(msg, "velocity")
        if gripper is not None and gripper_dq is None:
            gripper_dq = self._estimate_gripper_velocity(msg, gripper)
        with self.lock:
            if arm is not None:
                self.latest_arm_q = arm
            if arm_dq is not None:
                self.latest_arm_dq = arm_dq
            if gripper is not None:
                self.latest_gripper_q = gripper
            if gripper_dq is not None:
                self.latest_gripper_dq = gripper_dq

    # ------------------------------------------------------------------
    # Replay callbacks and recording
    # ------------------------------------------------------------------
    def replay_started_cb(self, msg: Bool):
        if not msg.data or self.state != self.STATE_WAITING_FOR_START:
            return
        self.replay_t0 = time.time()
        self.recording = True
        self.state = self.STATE_REPLAYING
        self.started_event.set()
        self._reset_record_buffers()
        self._schedule_gripper_events()
        self.get_logger().info("EE replay started; recording FK_state")

    def replay_finished_cb(self, msg: Bool):
        if not msg.data or self.state not in (
            self.STATE_WAITING_FOR_START,
            self.STATE_REPLAYING,
        ):
            return
        self.recording = False
        self._cancel_scheduled_events()
        self._save_fk_state()
        self.state = self.STATE_IDLE
        self.finished_event.set()
        self.get_logger().info("EE replay finished")

    def record_tick(self):
        if not self.recording or self.replay_t0 is None:
            return
        with self.lock:
            tcp_pose = list(self.latest_tcp_pose) if self.latest_tcp_pose else None
            arm_q = list(self.latest_arm_q) if self.latest_arm_q else None
            arm_dq = list(self.latest_arm_dq) if self.latest_arm_dq else None
            gripper_q = (
                list(self.latest_gripper_q)
                if self.latest_gripper_q is not None
                else [float("nan"), float("nan")]
            )
            gripper_dq = (
                list(self.latest_gripper_dq)
                if self.latest_gripper_dq is not None
                else [0.0, 0.0]
            )
        t_rel = time.time() - self.replay_t0
        if tcp_pose is not None:
            self.tcp_times.append(t_rel)
            self.tcp_poses.append(tcp_pose)
        if arm_q is not None:
            self.q_times.append(t_rel)
            self.q_values.append(arm_q + gripper_q)
        if arm_dq is not None:
            self.dq_times.append(t_rel)
            self.dq_values.append(arm_dq + gripper_dq)

    # ------------------------------------------------------------------
    # Main trajectory flow
    # ------------------------------------------------------------------
    def run_trajectory(self, traj_dir):
        pose_path = find_pose_file(traj_dir)
        times, poses = self._load_pose_file(pose_path)
        events = self._load_gripper_events(traj_dir)
        msg = self._build_trajectory_msg(times, poses)

        self.current_traj_dir = traj_dir
        self.current_events = events
        self.started_event.clear()
        self.finished_event.clear()
        self._reset_record_buffers()
        self.state = self.STATE_WAITING_FOR_START

        self.traj_pub.publish(msg)
        self.get_logger().info(
            f"Published {len(msg.points)} EE poses for {os.path.basename(traj_dir)} "
            f"(duration {times[-1] - times[0]:.2f}s, gripper events {len(events)})"
        )
        time.sleep(float(self.args.start_delay))
        self._publish_mode("replay")

        timeout = float(self.args.finish_timeout)
        timeout = None if timeout <= 0.0 else timeout
        if not self.finished_event.wait(timeout=timeout):
            self.recording = False
            self._cancel_scheduled_events()
            self._publish_mode("hold")
            self.state = self.STATE_IDLE
            raise TimeoutError(f"Timed out waiting for replay finish: {traj_dir}")

    def _publish_mode(self, mode):
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)
        self.get_logger().info(f"Published mode: {mode}")

    # ------------------------------------------------------------------
    # Pose file parsing
    # ------------------------------------------------------------------
    def _load_pose_file(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as data:
            keys = list(data.keys())
            if "timestamps" in data:
                times = np.asarray(data["timestamps"], dtype=float)
            elif "relative_times" in data:
                times = np.asarray(data["relative_times"], dtype=float)
            else:
                times = None

            pose_key = self.args.pose_key
            if pose_key:
                if pose_key not in data:
                    raise KeyError(f"{path} does not contain pose key '{pose_key}'")
            else:
                pose_key = self._guess_pose_key(data, keys)
            poses = np.asarray(data[pose_key], dtype=float)

        poses = self._normalize_pose_array(poses)
        if times is None:
            times = np.arange(poses.shape[0], dtype=float) / self.input_rate
        if times.shape[0] != poses.shape[0]:
            raise ValueError(
                f"{path}: timestamps length {times.shape[0]} != poses length {poses.shape[0]}"
            )

        times = times - times[0]
        poses = self._smooth_positions(poses)
        times, poses = self._drop_duplicate_times(times, poses)
        if poses.shape[0] < 2:
            raise ValueError(f"{path}: need at least two unique pose samples")
        return times, poses

    def _guess_pose_key(self, data, keys):
        for key in DEFAULT_POSE_KEYS:
            if key in data:
                return key
        for key in keys:
            if key in ("timestamps", "relative_times", "actions"):
                continue
            arr = np.asarray(data[key])
            if arr.ndim >= 2 and (7 in arr.shape or arr.shape[-2:] == (4, 4)):
                return key
        raise KeyError(f"Could not find a pose array key in npz keys: {keys}")

    def _normalize_pose_array(self, poses):
        if poses.ndim == 3 and poses.shape[1:] == (4, 4):
            out = np.zeros((poses.shape[0], 7), dtype=float)
            out[:, 0:3] = poses[:, 0:3, 3]
            for i, mat in enumerate(poses):
                out[i, 3:7] = self._quat_from_rotation_matrix(mat[0:3, 0:3])
            poses = out
        elif poses.ndim == 2 and poses.shape[1] == 16:
            mats = np.asarray([row.reshape((4, 4), order="F") for row in poses])
            poses = self._normalize_pose_array(mats)
        elif poses.ndim == 2 and poses.shape[0] == 7 and poses.shape[1] != 7:
            poses = poses.T
        elif poses.ndim != 2 or poses.shape[1] != 7:
            raise ValueError(f"Expected pose array shape (N,7), (N,16), or (N,4,4); got {poses.shape}")

        poses = np.asarray(poses, dtype=float)
        quat = poses[:, 3:7]
        norms = np.linalg.norm(quat, axis=1)
        if np.any(norms <= 1e-9):
            raise ValueError("Pose file contains a near-zero quaternion")
        poses[:, 3:7] = quat / norms[:, None]
        for i in range(1, poses.shape[0]):
            if float(np.dot(poses[i - 1, 3:7], poses[i, 3:7])) < 0.0:
                poses[i, 3:7] *= -1.0
        return poses

    @staticmethod
    def _quat_from_rotation_matrix(rot):
        # Returns [x, y, z, w].
        trace = float(np.trace(rot))
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot[2, 1] - rot[1, 2]) * s
            y = (rot[0, 2] - rot[2, 0]) * s
            z = (rot[1, 0] - rot[0, 1]) * s
        else:
            diag = np.diag(rot)
            idx = int(np.argmax(diag))
            if idx == 0:
                s = 2.0 * np.sqrt(max(0.0, 1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]))
                w = (rot[2, 1] - rot[1, 2]) / s
                x = 0.25 * s
                y = (rot[0, 1] + rot[1, 0]) / s
                z = (rot[0, 2] + rot[2, 0]) / s
            elif idx == 1:
                s = 2.0 * np.sqrt(max(0.0, 1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]))
                w = (rot[0, 2] - rot[2, 0]) / s
                x = (rot[0, 1] + rot[1, 0]) / s
                y = 0.25 * s
                z = (rot[1, 2] + rot[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(max(0.0, 1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]))
                w = (rot[1, 0] - rot[0, 1]) / s
                x = (rot[0, 2] + rot[2, 0]) / s
                y = (rot[1, 2] + rot[2, 1]) / s
                z = 0.25 * s
        quat = np.asarray([x, y, z, w], dtype=float)
        return quat / np.linalg.norm(quat)

    def _smooth_positions(self, poses):
        window = self.trajectory_smoothing_window
        if window <= 1 or poses.shape[0] < 3:
            return poses
        if window % 2 == 0:
            window += 1
        window = min(window, poses.shape[0] if poses.shape[0] % 2 == 1 else poses.shape[0] - 1)
        if window <= 1:
            return poses

        smoothed = poses.copy()
        pad = window // 2
        kernel = np.ones(window, dtype=float) / float(window)
        padded = np.pad(poses[:, 0:3], ((pad, pad), (0, 0)), mode="edge")
        for axis in range(3):
            smoothed[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
        smoothed[0, 0:3] = poses[0, 0:3]
        smoothed[-1, 0:3] = poses[-1, 0:3]
        self.get_logger().info(f"Applied EE position smoothing window: {window} samples")
        return smoothed

    @staticmethod
    def _drop_duplicate_times(times, poses):
        keep = []
        last = -1.0
        for i, t in enumerate(times):
            if float(t) > last + 1e-6:
                keep.append(i)
                last = float(t)
        return times[keep], poses[keep, :]

    def _build_trajectory_msg(self, times, poses):
        msg = MultiDOFJointTrajectory()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.joint_names = [self.args.ee_joint_name]

        velocities = self._linear_velocities(times, poses[:, 0:3])
        t0 = float(times[0])
        for t, pose, velocity in zip(times, poses, velocities):
            point = MultiDOFJointTrajectoryPoint()
            transform = Transform()
            transform.translation.x = float(pose[0])
            transform.translation.y = float(pose[1])
            transform.translation.z = float(pose[2])
            transform.rotation.x = float(pose[3])
            transform.rotation.y = float(pose[4])
            transform.rotation.z = float(pose[5])
            transform.rotation.w = float(pose[6])
            point.transforms.append(transform)

            twist = Twist()
            twist.linear.x = float(velocity[0])
            twist.linear.y = float(velocity[1])
            twist.linear.z = float(velocity[2])
            point.velocities.append(twist)
            point.time_from_start = duration_from_seconds(float(t - t0))
            msg.points.append(point)
        return msg

    @staticmethod
    def _linear_velocities(times, positions):
        velocities = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            if i == 0:
                dt = times[1] - times[0]
                dp = positions[1] - positions[0]
            elif i + 1 == positions.shape[0]:
                dt = times[i] - times[i - 1]
                dp = positions[i] - positions[i - 1]
            else:
                dt = times[i + 1] - times[i - 1]
                dp = positions[i + 1] - positions[i - 1]
            if dt > 1e-9:
                velocities[i, :] = dp / dt
        return velocities

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------
    def _load_gripper_events(self, traj_dir):
        if self.args.gripper_source == "none":
            return []

        candidates = []
        if self.args.gripper_source in ("auto", "replay"):
            candidates.append(os.path.join(traj_dir, "replay", "gripper_events.npz"))
        if self.args.gripper_source in ("auto", "teach"):
            candidates.append(os.path.join(traj_dir, "teach", "gripper_events.npz"))

        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            self.get_logger().warn(f"No gripper_events.npz found for {traj_dir}")
            return []

        with np.load(path, allow_pickle=True) as data:
            if "relative_times" in data:
                times = np.asarray(data["relative_times"], dtype=float)
            elif "timestamps" in data:
                times = np.asarray(data["timestamps"], dtype=float)
            else:
                return []
            actions = [stringify_action(a).lower() for a in data["actions"]]

        events = []
        for t, action in zip(times, actions):
            if action in ("open", "close", "home"):
                events.append((max(0.0, float(t)), action))
        events.sort(key=lambda item: item[0])
        return events

    def _schedule_gripper_events(self):
        self._cancel_scheduled_events()
        for ev_t, ev_action in self.current_events:
            timer = threading.Timer(ev_t, self._fire_gripper_event, args=(ev_action,))
            timer.daemon = True
            timer.start()
            self.scheduled_event_timers.append(timer)
        if self.current_events:
            self.get_logger().info(f"Scheduled {len(self.current_events)} gripper events")

    def _fire_gripper_event(self, action):
        if self.state != self.STATE_REPLAYING:
            return
        self._send_gripper(action)
        if self.replay_t0 is not None:
            self.gripper_event_log.append((time.time() - self.replay_t0, action))

    def _cancel_scheduled_events(self):
        for timer in self.scheduled_event_timers:
            timer.cancel()
        self.scheduled_event_timers = []

    def _send_gripper(self, action):
        if action == "open":
            client = self.move_client
            goal = Move.Goal()
            goal.width = float(self.args.open_width)
            goal.speed = float(self.args.gripper_speed)
        elif action == "close":
            client = self.grasp_client
            goal = Grasp.Goal()
            goal.width = float(self.args.close_width)
            goal.speed = float(self.args.gripper_speed)
            goal.force = float(self.args.gripper_force)
            goal.epsilon.inner = float(self.args.grasp_epsilon)
            goal.epsilon.outer = float(self.args.grasp_epsilon)
        elif action == "home":
            client = self.homing_client
            goal = Homing.Goal()
        else:
            return

        if not client.server_is_ready():
            client.wait_for_server(timeout_sec=0.2)
        client.send_goal_async(goal).add_done_callback(
            self._action_done_cb_factory(action)
        )

    def _action_done_cb_factory(self, label):
        def cb(future):
            try:
                handle = future.result()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"{label}: send_goal failed: {exc}")
                return
            if not handle.accepted:
                self.get_logger().warn(f"{label}: rejected by gripper server")
                return
            handle.get_result_async().add_done_callback(
                lambda f: self._on_gripper_result(label, f)
            )
        return cb

    def _on_gripper_result(self, label, future):
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"{label}: result fetch failed: {exc}")
            return
        success = getattr(result.result, "success", None)
        if result.status != GoalStatus.STATUS_SUCCEEDED and not success:
            self.get_logger().warn(f"{label}: status={result.status} success={success}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _reset_record_buffers(self):
        self.tcp_times = []
        self.tcp_poses = []
        self.q_times = []
        self.q_values = []
        self.dq_times = []
        self.dq_values = []
        self.gripper_event_log = []

    def _save_fk_state(self):
        if self.current_traj_dir is None:
            return
        state_dir = os.path.join(self.current_traj_dir, "FK_state")
        os.makedirs(state_dir, exist_ok=True)

        self._save_npz(
            os.path.join(state_dir, "tcp_pose.npz"),
            "tcp_pose",
            self.tcp_times,
            self.tcp_poses,
            7,
        )
        self._save_npz(
            os.path.join(state_dir, "joint_velocity.npz"),
            "joint_velocity",
            self.dq_times,
            self.dq_values,
            9,
        )
        self._save_npz(
            os.path.join(state_dir, "joint_pos.npz"),
            "joint_pos",
            self.q_times,
            self.q_values,
            9,
        )

        ev_path = os.path.join(state_dir, "gripper_events.npz")
        if self.gripper_event_log:
            ev_t = np.asarray([e[0] for e in self.gripper_event_log], dtype=float)
            ev_a = np.asarray([e[1] for e in self.gripper_event_log])
        else:
            ev_t = np.zeros((0,), dtype=float)
            ev_a = np.zeros((0,), dtype=object)
        np.savez(ev_path, relative_times=ev_t, actions=ev_a)

    def _save_npz(self, path, key, times, values, width):
        if values:
            arr = np.asarray(values, dtype=float)
            ts = np.asarray(times, dtype=float)
        else:
            arr = np.zeros((0, width), dtype=float)
            ts = np.zeros((0,), dtype=float)
            self.get_logger().warn(f"No samples captured for {key}")
        np.savez(path, timestamps=ts, **{key: arr})
        self.get_logger().info(f"Wrote {path}")

    def shutdown(self):
        self.recording = False
        self._cancel_scheduled_events()
        if self.state != self.STATE_IDLE:
            self._publish_mode("hold")


def discover_trajectories(data_dir):
    trajectories = []
    for name in os.listdir(data_dir):
        match = TRAJ_DIR_PATTERN.match(name)
        if not match:
            continue
        traj_dir = os.path.join(data_dir, name)
        if os.path.isdir(traj_dir) and find_pose_file(traj_dir, required=False):
            trajectories.append((int(match.group(1)), traj_dir))
    return [path for _, path in sorted(trajectories)]


def trajectory_number(traj_dir):
    name = os.path.basename(os.path.normpath(traj_dir))
    match = TRAJ_DIR_PATTERN.match(name)
    if not match:
        raise ValueError(f"Invalid trajectory directory name: {traj_dir}")
    return int(match.group(1))


def filter_trajectories_from(trajectories, start_num):
    return [
        traj_dir for traj_dir in trajectories
        if trajectory_number(traj_dir) >= start_num
    ]


def choose_start_trajectory(trajectories, start_num_arg):
    available_nums = [trajectory_number(path) for path in trajectories]
    min_num = min(available_nums)
    max_num = max(available_nums)

    if start_num_arg is not None:
        selected = filter_trajectories_from(trajectories, int(start_num_arg))
        if not selected:
            raise ValueError(
                f"No traj_N with N >= {start_num_arg}; "
                f"available range is traj_{min_num}..traj_{max_num}"
            )
        return selected

    while True:
        raw = input("starting from traj num? ").strip()
        try:
            start_num = int(raw)
        except ValueError:
            print("Please enter an integer trajectory number.")
            continue

        selected = filter_trajectories_from(trajectories, start_num)
        if selected:
            first = trajectory_number(selected[0])
            last = trajectory_number(selected[-1])
            print(
                f"Will replay {len(selected)} trajectory(s), "
                f"from traj_{first} through traj_{last}."
            )
            return selected
        print(
            f"No traj_N with N >= {start_num}; "
            f"available range is traj_{min_num}..traj_{max_num}."
        )


def find_pose_file(traj_dir, required=True):
    replay_dir = os.path.join(traj_dir, "replay")
    for filename in POSE_FILENAMES:
        path = os.path.join(replay_dir, filename)
        if os.path.isfile(path):
            return path
    if required:
        raise FileNotFoundError(os.path.join(replay_dir, POSE_FILENAMES[0]))
    return None


def prompt_data_dir(default):
    if default:
        return os.path.expanduser(default)
    while True:
        value = input("enter data folder path: ").strip()
        if value:
            return os.path.expanduser(value)


def existing_fk_outputs(traj_dir):
    state_dir = os.path.join(traj_dir, "FK_state")
    names = (
        "tcp_pose.npz",
        "joint_velocity.npz",
        "joint_pos.npz",
        "gripper_events.npz",
    )
    return [os.path.join(state_dir, name) for name in names
            if os.path.exists(os.path.join(state_dir, name))]


def confirm_overwrite_if_needed(trajectories, overwrite):
    existing = [(traj_dir, existing_fk_outputs(traj_dir)) for traj_dir in trajectories]
    existing = [(traj_dir, paths) for traj_dir, paths in existing if paths]
    if not existing:
        return
    if overwrite:
        print(
            f"--overwrite set: will overwrite existing FK_state outputs for "
            f"{len(existing)} trajectory(s)."
        )
        return

    print("WARNING: replay will overwrite existing FK_state outputs.")
    print(f"Affected trajectories: {len(existing)}")
    for traj_dir, paths in existing[:10]:
        names = ", ".join(os.path.basename(path) for path in paths)
        print(f"  {os.path.basename(traj_dir)}: {names}")
    if len(existing) > 10:
        print(f"  ... {len(existing) - 10} more")
    answer = input("Type 'yes' to continue and overwrite, or anything else to abort: ")
    if answer.strip().lower() != "yes":
        raise RuntimeError("Aborted before replay to avoid overwriting existing FK_state data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, default="",
                        help="Folder containing traj_N directories")
    parser.add_argument("--test", action="store_true",
                        help="Replay one trajectory per keypress")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing traj_N/FK_state outputs without prompting")
    parser.add_argument("--start_traj", type=int, default=None,
                        help="Start replay from this traj_N number; default prompts")
    parser.add_argument("--record_rate", type=float, default=100.0,
                        help="FK_state recording rate in Hz")
    parser.add_argument("--input_rate", type=float, default=100.0,
                        help="Fallback Hz if EE_pose_FK.npz has no timestamps")
    parser.add_argument("--trajectory_smoothing_window", type=int, default=5,
                        help="Odd moving-average window for EE positions; <=1 disables")
    parser.add_argument("--pose_key", type=str, default="",
                        help="NPZ key for the (N,7) pose array; default auto-detect")
    parser.add_argument("--frame_id", type=str, default="fr3_link0")
    parser.add_argument("--ee_joint_name", type=str, default="fr3_hand_tcp")
    parser.add_argument("--start_delay", type=float, default=0.2)
    parser.add_argument("--finish_timeout", type=float, default=0.0,
                        help="Seconds to wait for finish; <=0 waits indefinitely")
    parser.add_argument("--mode_topic", type=str, default="/ee_pose_replay/mode")
    parser.add_argument("--trajectory_topic", type=str,
                        default="/ee_pose_replay/trajectory")
    parser.add_argument("--start_topic", type=str,
                        default="/ee_pose_replay/replay_started")
    parser.add_argument("--finish_topic", type=str,
                        default="/ee_pose_replay/replay_finished")
    parser.add_argument("--robot_state_topic", type=str,
                        default="/franka_robot_state_broadcaster/robot_state")
    parser.add_argument("--joint_state_topic", type=str, default="/joint_states")
    parser.add_argument("--gripper_source", choices=("auto", "replay", "teach", "none"),
                        default="auto")
    parser.add_argument("--gripper_move_action", type=str, default="/franka_gripper/move")
    parser.add_argument("--gripper_grasp_action", type=str, default="/franka_gripper/grasp")
    parser.add_argument("--gripper_homing_action", type=str, default="/franka_gripper/homing")
    parser.add_argument("--open_width", type=float, default=0.08)
    parser.add_argument("--close_width", type=float, default=0.0)
    parser.add_argument("--gripper_speed", type=float, default=0.1)
    parser.add_argument("--gripper_force", type=float, default=20.0)
    parser.add_argument("--grasp_epsilon", type=float, default=0.08)
    args = parser.parse_args()

    data_dir = prompt_data_dir(args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(data_dir)
    trajectories = discover_trajectories(data_dir)
    if not trajectories:
        raise FileNotFoundError(f"No traj_N/replay/EE_pose_FK.npz found under {data_dir}")
    trajectories = choose_start_trajectory(trajectories, args.start_traj)
    confirm_overwrite_if_needed(trajectories, args.overwrite)

    rclpy.init(args=sys.argv)
    node = EeControlReplay(args)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        for index, traj_dir in enumerate(trajectories):
            if args.test:
                choice = input(
                    f"[test] Press Enter to replay {os.path.basename(traj_dir)} "
                    "or q to quit: "
                ).strip().lower()
                if choice == "q":
                    break
            node.run_trajectory(traj_dir)
            if not args.test:
                continue
            if index + 1 < len(trajectories):
                print("Replay saved. Next trajectory is waiting for your key.")
        node.get_logger().info("EE control replay batch complete")
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
