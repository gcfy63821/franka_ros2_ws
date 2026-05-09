#!/usr/bin/env python3
"""Teach-and-replay orchestrator for the TeachReplayController.

Workflow (single keyboard-driven session):

    IDLE --[t]--> TEACHING
        Robot is in TEACH mode (zero torques). Drag it through the desired
        trajectory. While teaching, press 'o' / 'c' to open / close the
        gripper. Each gripper press is logged with a timestamp.

    TEACHING --[s]--> READY
        Teach trajectory and gripper events are saved to disk.

    READY --[r]--> REPLAYING
        Recorded trajectory is published to the controller. Mode is switched
        to REPLAY. When the controller emits /teach_replay/replay_started,
        we start the video writer and the joint-state recorder, and we
        schedule timers that re-issue gripper open/close actions at the
        recorded relative timestamps. /teach_replay/replay_finished closes
        the video writer and saves the replay-side data.

    REPLAYING --> READY (auto on finish)

    any --[q]--> quit

Files (per session, under <output_dir>/session_YYYYMMDD_HHMMSS/):
    teach/joint_trajectory.npz         (timestamps, joint_positions, gripper_widths)
    teach/gripper_events.npz           (relative_times, actions)
    replay/recording.avi               (video, optional)
    replay/joint_trajectory.npz        (executed joint positions)
    replay/gripper_events.npz          (gripper action issue timestamps)

Usage:
    python3 teach_replay_orchestrator.py [--fps 30] [--record_rate 100]
        [--output_dir ~/robot_recordings]
        [--image_topic /camera/camera/color/image_raw]
        [--trajectory_smoothing_window 11]
        [--no_video]
"""

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Header, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    import cv2
    from cv_bridge import CvBridge
    HAVE_CV = True
except ImportError:
    HAVE_CV = False

# Action types are looked up lazily so that a missing franka_msgs build does
# not prevent the script from at least starting and reporting the issue.
try:
    from franka_msgs.action import Grasp, Homing, Move
    HAVE_GRIPPER_ACTIONS = True
except ImportError:
    HAVE_GRIPPER_ACTIONS = False


ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]


class TeachReplayOrchestrator(Node):
    STATE_IDLE = "IDLE"
    STATE_TEACHING = "TEACHING"
    STATE_READY = "READY"
    STATE_REPLAYING = "REPLAYING"

    def __init__(self, args):
        super().__init__("teach_replay_orchestrator")
        self.fps = args.fps
        self.record_rate = args.record_rate
        self.trajectory_smoothing_window = args.trajectory_smoothing_window
        self.output_dir = os.path.expanduser(args.output_dir)
        self.image_topic = args.image_topic
        self.record_video = (not args.no_video) and HAVE_CV
        os.makedirs(self.output_dir, exist_ok=True)

        # Session state
        self.state = self.STATE_IDLE
        self.session_dir = None
        self.lock = threading.Lock()

        # Joint state cache
        self.latest_arm_q = None      # 7-vector
        self.latest_gripper = None    # (finger1, finger2)

        # Teach buffers
        self.teach_t0 = None
        self.teach_times = []         # relative seconds
        self.teach_q = []             # arm joint positions
        self.teach_gripper = []       # gripper finger pair
        self.teach_events = []        # list of (rel_time, action)

        # Replay buffers
        self.replay_t0 = None
        self.replay_times = []
        self.replay_q = []
        self.replay_event_log = []    # (rel_time_at_call, action)
        self.replay_video_path = None
        self.video_writer = None
        self.video_lock = threading.Lock()
        self.scheduled_event_timers = []
        self.recorded_trajectory = None  # tuple (times, q, events) saved after teach

        # Image
        self.cv_bridge = CvBridge() if HAVE_CV else None
        self.latest_image = None
        self.image_lock = threading.Lock()

        # ROS interfaces
        qos = 10
        self.mode_pub = self.create_publisher(String, "/teach_replay/mode", qos)
        self.traj_pub = self.create_publisher(
            JointTrajectory, "/teach_replay/trajectory", qos
        )
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_cb, qos
        )
        self.replay_started_sub = self.create_subscription(
            Bool, "/teach_replay/replay_started", self.replay_started_cb, qos
        )
        self.replay_finished_sub = self.create_subscription(
            Bool, "/teach_replay/replay_finished", self.replay_finished_cb, qos
        )

        if self.record_video:
            self.image_sub = self.create_subscription(
                Image, self.image_topic, self.image_cb, qos_profile_sensor_data
            )
            self.get_logger().info(f"Video enabled, image_topic={self.image_topic}")
        else:
            if not HAVE_CV and not args.no_video:
                self.get_logger().warn(
                    "cv2 / cv_bridge not available; video recording disabled"
                )

        # Gripper action clients
        self.move_client = None
        self.grasp_client = None
        self.homing_client = None
        if HAVE_GRIPPER_ACTIONS:
            self.move_client = ActionClient(self, Move, "/franka_gripper/move")
            self.grasp_client = ActionClient(self, Grasp, "/franka_gripper/grasp")
            self.homing_client = ActionClient(self, Homing, "/franka_gripper/homing")
            self.get_logger().info("Gripper action clients created")
        else:
            self.get_logger().warn(
                "franka_msgs not importable; gripper open/close keys disabled"
            )

        # Recording timer (active only during TEACHING / REPLAYING)
        self.record_timer = self.create_timer(
            1.0 / self.record_rate, self.record_tick
        )

        # Keyboard thread
        self.keyboard_thread = threading.Thread(
            target=self.keyboard_loop, daemon=True
        )
        self.keyboard_thread.start()

        self.print_help()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def print_help(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info("Teach-Replay Orchestrator. Keys:")
        self.get_logger().info("  t : start teaching (drag the robot)")
        self.get_logger().info("  o : gripper OPEN  (during teach)")
        self.get_logger().info("  c : gripper CLOSE (during teach)")
        self.get_logger().info("  s : stop teaching, save trajectory")
        self.get_logger().info("  r : replay last teach + record video/joints")
        self.get_logger().info("  h : home gripper")
        self.get_logger().info("  q : quit")
        self.get_logger().info("=" * 60)

    def transition(self, new_state):
        old = self.state
        self.state = new_state
        self.get_logger().info(f"State: {old} -> {new_state}")

    def ensure_session_dir(self):
        if self.session_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(self.output_dir, f"session_{ts}")
            os.makedirs(os.path.join(self.session_dir, "teach"), exist_ok=True)
            os.makedirs(os.path.join(self.session_dir, "replay"), exist_ok=True)
            self.get_logger().info(f"Session dir: {self.session_dir}")
        return self.session_dir

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------
    def joint_state_cb(self, msg: JointState):
        # /joint_states aggregates arm + gripper. Pull arm joints by name to
        # avoid index assumptions; fall back to first 7 if names absent.
        arm = [None] * 7
        gripper = [None, None]
        if msg.name:
            for i, name in enumerate(msg.name):
                for j, target in enumerate(ARM_JOINT_NAMES):
                    if name == target and i < len(msg.position):
                        arm[j] = msg.position[i]
                if "finger_joint1" in name and i < len(msg.position):
                    gripper[0] = msg.position[i]
                elif "finger_joint2" in name and i < len(msg.position):
                    gripper[1] = msg.position[i]
        else:
            if len(msg.position) >= 7:
                arm = list(msg.position[:7])
            if len(msg.position) >= 9:
                gripper = [msg.position[7], msg.position[8]]

        with self.lock:
            if all(v is not None for v in arm):
                self.latest_arm_q = arm
            if all(v is not None for v in gripper):
                self.latest_gripper = gripper

    def image_cb(self, msg: Image):
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"image conversion failed: {exc}")
            return
        with self.image_lock:
            self.latest_image = frame
        # Write a frame if video active. We push frames at image rate so the
        # output video runs at ~camera fps regardless of fps arg.
        if self.video_writer is not None:
            with self.video_lock:
                if self.video_writer is not None:
                    self.video_writer.write(frame)

    def replay_started_cb(self, msg: Bool):
        if not msg.data:
            return
        if self.state != self.STATE_REPLAYING:
            self.get_logger().warn(
                f"replay_started received but state={self.state}; ignoring"
            )
            return
        self.replay_t0 = time.time()
        self.replay_times = []
        self.replay_q = []
        self.replay_event_log = []
        self.get_logger().info("Replay started; recording joints/video")

        if self.record_video and self.latest_image is not None:
            self._open_video_writer()

        # Schedule gripper events at their recorded relative timestamps.
        self.scheduled_event_timers = []
        if self.recorded_trajectory is not None:
            _, _, events = self.recorded_trajectory
            for ev_t, ev_action in events:
                timer = threading.Timer(
                    ev_t, self._fire_gripper_event, args=(ev_t, ev_action)
                )
                timer.daemon = True
                timer.start()
                self.scheduled_event_timers.append(timer)
            self.get_logger().info(
                f"Scheduled {len(events)} gripper events for replay"
            )

    def replay_finished_cb(self, msg: Bool):
        if not msg.data:
            return
        if self.state != self.STATE_REPLAYING:
            return
        self.get_logger().info("Replay finished; saving data")
        self._cancel_scheduled_events()
        self._close_video_writer()
        self._save_replay_data()
        self.transition(self.STATE_READY)

    # ------------------------------------------------------------------
    # Recording timer
    # ------------------------------------------------------------------
    def record_tick(self):
        with self.lock:
            arm = list(self.latest_arm_q) if self.latest_arm_q else None
            grip = list(self.latest_gripper) if self.latest_gripper else None
        now = time.time()

        if self.state == self.STATE_TEACHING and arm is not None:
            if self.teach_t0 is None:
                self.teach_t0 = now
            self.teach_times.append(now - self.teach_t0)
            self.teach_q.append(arm)
            self.teach_gripper.append(grip if grip else [float("nan"), float("nan")])
        elif self.state == self.STATE_REPLAYING and self.replay_t0 is not None and arm is not None:
            self.replay_times.append(now - self.replay_t0)
            self.replay_q.append(arm)

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------
    def keyboard_loop(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key in ("\x03", "q", "Q"):
                        self.get_logger().info("Quit requested")
                        rclpy.shutdown()
                        return
                    self._handle_key(key.lower())
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key):
        if key == "t":
            self.cmd_start_teach()
        elif key == "s":
            self.cmd_stop_teach()
        elif key == "r":
            self.cmd_replay()
        elif key == "o":
            self.cmd_gripper("open")
        elif key == "c":
            self.cmd_gripper("close")
        elif key == "h":
            self.cmd_gripper("home")
        else:
            # ignore other keys silently to avoid log spam
            return

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    def cmd_start_teach(self):
        if self.state != self.STATE_IDLE and self.state != self.STATE_READY:
            self.get_logger().warn(f"Cannot start teach in state {self.state}")
            return
        # Reset session: a fresh teach starts a new session folder so old
        # data is not overwritten.
        self.session_dir = None
        self.ensure_session_dir()
        self.teach_t0 = None
        self.teach_times = []
        self.teach_q = []
        self.teach_gripper = []
        self.teach_events = []
        self.recorded_trajectory = None
        # Make sure controller is in TEACH (zero torque).
        self._publish_mode("teach")
        self.transition(self.STATE_TEACHING)
        self.get_logger().info("TEACH started: drag the robot. 'o'/'c' for gripper, 's' to stop.")

    def cmd_stop_teach(self):
        if self.state != self.STATE_TEACHING:
            self.get_logger().warn(f"Cannot stop teach in state {self.state}")
            return
        if not self.teach_times:
            self.get_logger().warn("No teach samples captured; staying in TEACHING")
            return
        self._save_teach_data()
        # Pre-bake the trajectory message for later replay.
        times = np.asarray(self.teach_times)
        qs = np.asarray(self.teach_q)
        events = list(self.teach_events)
        self.recorded_trajectory = (times, qs, events)
        self.transition(self.STATE_READY)
        self.get_logger().info(
            f"Teach saved: {len(times)} samples, duration {times[-1]:.2f}s, "
            f"{len(events)} gripper events. Press 'r' to replay."
        )

    def cmd_replay(self):
        if self.state != self.STATE_READY:
            self.get_logger().warn(
                f"Cannot replay in state {self.state} (run teach + 's' first)"
            )
            return
        if self.recorded_trajectory is None:
            self.get_logger().warn("No recorded trajectory available")
            return
        times, qs, _ = self.recorded_trajectory
        msg = self._build_trajectory_msg(times, qs)
        self.traj_pub.publish(msg)
        self.get_logger().info(f"Published trajectory ({len(times)} points)")
        # Short delay so the controller registers the trajectory before
        # the mode switch flips it into REPLAY.
        threading.Timer(0.2, lambda: self._publish_mode("replay")).start()
        self.transition(self.STATE_REPLAYING)
        self.get_logger().info(
            "Waiting for controller pre-roll (move-to-start). Recording will "
            "begin when /teach_replay/replay_started fires."
        )

    def cmd_gripper(self, action):
        if action == "open":
            self._send_move(width=0.08, speed=0.1)
            label = "open"
        elif action == "close":
            self._send_grasp(width=0.0, speed=0.1, force=20.0)
            label = "close"
        elif action == "home":
            self._send_homing()
            label = "home"
        else:
            return

        # Log gripper events relative to the current phase.
        now = time.time()
        if self.state == self.STATE_TEACHING and self.teach_t0 is not None:
            t_rel = now - self.teach_t0
            self.teach_events.append((t_rel, label))
            self.get_logger().info(f"Teach: gripper {label} @ t={t_rel:.2f}s")
        elif self.state == self.STATE_REPLAYING and self.replay_t0 is not None:
            t_rel = now - self.replay_t0
            self.replay_event_log.append((t_rel, label))
            self.get_logger().info(f"Replay: gripper {label} @ t={t_rel:.2f}s")
        else:
            self.get_logger().info(f"Gripper {label} (not recording)")

    # ------------------------------------------------------------------
    # Trajectory packaging
    # ------------------------------------------------------------------
    def _smooth_joint_trajectory(self, qs):
        window = int(self.trajectory_smoothing_window)
        if window <= 1 or len(qs) < 3:
            return qs
        if window % 2 == 0:
            window += 1
        window = min(window, len(qs) if len(qs) % 2 == 1 else len(qs) - 1)
        if window <= 1:
            return qs

        pad = window // 2
        kernel = np.ones(window, dtype=float) / float(window)
        padded = np.pad(qs, ((pad, pad), (0, 0)), mode="edge")
        smoothed = np.empty_like(qs, dtype=float)
        for joint_idx in range(qs.shape[1]):
            smoothed[:, joint_idx] = np.convolve(
                padded[:, joint_idx], kernel, mode="valid"
            )

        # Preserve exact endpoints so replay starts/finishes at the taught poses.
        smoothed[0, :] = qs[0, :]
        smoothed[-1, :] = qs[-1, :]
        self.get_logger().info(
            f"Applied joint trajectory smoothing window: {window} samples"
        )
        return smoothed

    def _build_trajectory_msg(self, times, qs):
        msg = JointTrajectory()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        msg.joint_names = list(ARM_JOINT_NAMES)
        qs = self._smooth_joint_trajectory(np.asarray(qs, dtype=float))
        # Make sure the first sample sits at t=0 so the controller has a
        # well-defined reference at replay start.
        t0 = times[0]
        for t, q in zip(times, qs):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in q]
            t_rel = float(t - t0)
            sec = int(t_rel)
            nsec = int(round((t_rel - sec) * 1e9))
            if nsec >= 1_000_000_000:
                sec += 1
                nsec -= 1_000_000_000
            pt.time_from_start = DurationMsg(sec=sec, nanosec=nsec)
            msg.points.append(pt)
        # Strict monotonicity: drop any duplicate-timestamp tail samples.
        cleaned = []
        last_t = -1.0
        for pt in msg.points:
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            if t > last_t + 1e-6:
                cleaned.append(pt)
                last_t = t
        msg.points = cleaned
        return msg

    # ------------------------------------------------------------------
    # Mode publishing
    # ------------------------------------------------------------------
    def _publish_mode(self, mode):
        m = String()
        m.data = mode
        self.mode_pub.publish(m)
        self.get_logger().info(f"Published mode: {mode}")

    # ------------------------------------------------------------------
    # Gripper action senders (fire-and-forget; do not block keyboard thread)
    # ------------------------------------------------------------------
    def _send_move(self, width, speed):
        if self.move_client is None:
            self.get_logger().warn("Move client unavailable")
            return
        if not self.move_client.server_is_ready():
            self.move_client.wait_for_server(timeout_sec=0.2)
        goal = Move.Goal()
        goal.width = float(width)
        goal.speed = float(speed)
        self.move_client.send_goal_async(goal).add_done_callback(
            self._action_done_cb_factory("move")
        )

    def _send_grasp(self, width, speed, force):
        if self.grasp_client is None:
            self.get_logger().warn("Grasp client unavailable")
            return
        if not self.grasp_client.server_is_ready():
            self.grasp_client.wait_for_server(timeout_sec=0.2)
        goal = Grasp.Goal()
        goal.width = float(width)
        goal.speed = float(speed)
        goal.force = float(force)
        # Generous epsilon so the grasp succeeds across a range of object widths.
        goal.epsilon.inner = 0.08
        goal.epsilon.outer = 0.08
        self.grasp_client.send_goal_async(goal).add_done_callback(
            self._action_done_cb_factory("grasp")
        )

    def _send_homing(self):
        if self.homing_client is None:
            self.get_logger().warn("Homing client unavailable")
            return
        if not self.homing_client.server_is_ready():
            self.homing_client.wait_for_server(timeout_sec=0.2)
        self.homing_client.send_goal_async(Homing.Goal()).add_done_callback(
            self._action_done_cb_factory("homing")
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
            res = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"{label}: result fetch failed: {exc}")
            return
        status_ok = res.status == GoalStatus.STATUS_SUCCEEDED
        success = getattr(res.result, "success", None)
        self.get_logger().info(
            f"{label}: status={res.status} success={success}"
        )
        if not status_ok and not success:
            self.get_logger().warn(f"{label} did not succeed")

    # ------------------------------------------------------------------
    # Replay-side scheduled gripper events
    # ------------------------------------------------------------------
    def _fire_gripper_event(self, scheduled_t, action):
        if self.state != self.STATE_REPLAYING:
            return
        self.cmd_gripper(action)

    def _cancel_scheduled_events(self):
        for t in self.scheduled_event_timers:
            t.cancel()
        self.scheduled_event_timers = []

    # ------------------------------------------------------------------
    # Video helpers
    # ------------------------------------------------------------------
    def _open_video_writer(self):
        with self.image_lock:
            frame = self.latest_image
        if frame is None:
            self.get_logger().warn("No image received yet; skipping video")
            return
        h, w = frame.shape[:2]
        path = os.path.join(self.session_dir, "replay", "recording.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        if not writer.isOpened():
            self.get_logger().error(f"Failed to open VideoWriter: {path}")
            return
        with self.video_lock:
            self.video_writer = writer
            self.replay_video_path = path
        self.get_logger().info(f"Video writer opened: {path} ({w}x{h}@{self.fps}fps)")

    def _close_video_writer(self):
        with self.video_lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                self.get_logger().info(f"Video saved: {self.replay_video_path}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_teach_data(self):
        path = os.path.join(self.session_dir, "teach", "joint_trajectory.npz")
        np.savez(
            path,
            timestamps=np.asarray(self.teach_times),
            joint_positions=np.asarray(self.teach_q),
            gripper_widths=np.asarray(self.teach_gripper),
        )
        self.get_logger().info(f"Wrote {path}")

        ev_path = os.path.join(self.session_dir, "teach", "gripper_events.npz")
        if self.teach_events:
            ev_t = np.asarray([e[0] for e in self.teach_events])
            ev_a = np.asarray([e[1] for e in self.teach_events])
        else:
            ev_t = np.zeros((0,))
            ev_a = np.zeros((0,), dtype=object)
        np.savez(ev_path, relative_times=ev_t, actions=ev_a)
        self.get_logger().info(f"Wrote {ev_path}")

    def _save_replay_data(self):
        if not self.replay_times:
            self.get_logger().warn("No replay samples captured")
            return
        path = os.path.join(self.session_dir, "replay", "joint_trajectory.npz")
        np.savez(
            path,
            timestamps=np.asarray(self.replay_times),
            joint_positions=np.asarray(self.replay_q),
        )
        self.get_logger().info(f"Wrote {path}")

        ev_path = os.path.join(self.session_dir, "replay", "gripper_events.npz")
        if self.replay_event_log:
            ev_t = np.asarray([e[0] for e in self.replay_event_log])
            ev_a = np.asarray([e[1] for e in self.replay_event_log])
        else:
            ev_t = np.zeros((0,))
            ev_a = np.zeros((0,), dtype=object)
        np.savez(ev_path, relative_times=ev_t, actions=ev_a)
        self.get_logger().info(f"Wrote {ev_path}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def shutdown(self):
        self._cancel_scheduled_events()
        self._close_video_writer()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Output video FPS (default: 30)")
    parser.add_argument("--record_rate", type=float, default=100.0,
                        help="Joint-state recording rate during teach/replay (Hz, default: 100)")
    parser.add_argument("--trajectory_smoothing_window", type=int, default=11,
                        help="Odd moving-average window for replay joint trajectory smoothing; "
                             "set <=1 to disable (default: 11 samples)")
    parser.add_argument("--output_dir", type=str,
                        default="~/robot_recordings",
                        help="Output directory (default: ~/robot_recordings)")
    parser.add_argument("--image_topic", type=str,
                        default="/camera/camera/color/image_raw",
                        help="Image topic for video recording")
    parser.add_argument("--no_video", action="store_true",
                        help="Disable video recording (joint-only mode)")
    args = parser.parse_args()

    rclpy.init(args=sys.argv)
    node = TeachReplayOrchestrator(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
