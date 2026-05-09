// Copyright (c) 2026 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "franka_example_controllers/visibility_control.h"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

// Single controller with two modes:
//   TEACH  : zero torques (gravity compensation), robot can be hand-guided
//   REPLAY : dense joint-trajectory tracking via joint impedance
// Mode is switched at runtime via /teach_replay/mode (std_msgs/String).
// A trajectory is supplied via /teach_replay/trajectory (trajectory_msgs/JointTrajectory)
// with time_from_start populated for every point. The controller linearly
// interpolates between adjacent points by elapsed time and tracks the
// interpolated reference with impedance control.
class TeachReplayController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;

  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;

  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;

 private:
  // Phase is the controller's internal state machine. Externally users still
  // toggle between "teach" and "replay" via the mode topic; the REPLAY
  // command internally walks PRE_ROLL -> TRACKING.
  enum class Phase : int { TEACH = 0, PRE_ROLL = 1, TRACKING = 2 };

  static constexpr int kNumJoints = 7;

  // Trajectory snapshot used by the realtime update loop.
  struct Trajectory {
    std::vector<double> times;                // seconds, monotonically increasing, times[0] == 0
    std::vector<Vector7d> positions;          // joint positions per point
    std::vector<Vector7d> velocities;         // feedforward dq, finite-differenced if not provided
  };

  void modeCallback(const std_msgs::msg::String::SharedPtr msg);
  void trajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void updateJointStates();
  void publishStarted();
  void publishFinished();

  // Parameters
  std::string arm_id_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  std::string mode_topic_;
  std::string trajectory_topic_;
  std::string start_topic_;
  std::string finish_topic_;
  // Pre-roll (move-to-start) configuration. When the user requests REPLAY,
  // the controller first runs a min-jerk move from the current pose to
  // trajectory.points[0] before tracking the recorded trajectory, so the
  // robot does not snap if the user has dragged it away from the start.
  bool move_to_start_enabled_{true};
  double move_to_start_min_duration_{4.0};   // seconds
  double move_to_start_max_velocity_{0.2};   // rad/s, peak per-joint cap

  // State
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d hold_q_;  // pose to hold after replay completes (last trajectory point)

  std::atomic<int> phase_{static_cast<int>(Phase::TEACH)};
  // Pending trajectory swap: callback fills next_trajectory_ + sets has_new_trajectory_,
  // realtime update() consumes it without blocking the callback.
  std::mutex trajectory_mutex_;
  std::shared_ptr<Trajectory> next_trajectory_;  // guarded by trajectory_mutex_
  std::atomic<bool> has_new_trajectory_{false};

  std::shared_ptr<Trajectory> active_trajectory_;  // owned by update() thread
  bool start_pending_{false};
  bool finish_pending_{false};
  double replay_elapsed_{0.0};

  // Pre-roll state (only valid while phase_ == PRE_ROLL).
  Vector7d pre_roll_q_start_;
  double pre_roll_duration_{0.0};
  double pre_roll_elapsed_{0.0};

  // Pub/sub
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mode_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_sub_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Bool>> start_pub_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Bool>> finish_pub_;
};

}  // namespace franka_example_controllers
