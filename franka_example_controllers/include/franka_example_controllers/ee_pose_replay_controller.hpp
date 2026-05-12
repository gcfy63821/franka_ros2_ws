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

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include "franka_example_controllers/visibility_control.h"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

// Replays end-effector poses with a Cartesian impedance controller.
//
// The controller follows the same external protocol as the previous version
// (mode topic + MultiDOFJointTrajectory + start/finish booleans), but the
// inner loop is torque-based: it reads joint states + Jacobian from the
// franka_robot_model interface, computes the Cartesian pose error against a
// smooth reference (HOLD / PRE_ROLL min-jerk / TRACKING cubic Hermite), and
// emits joint efforts via tau = J^T (Kp*err + Kd*err_dot) + nullspace + coriolis.
class EePoseReplayController : public controller_interface::ControllerInterface {
 public:
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
  enum class Phase : int { HOLD = 0, PRE_ROLL = 1, TRACKING = 2 };

  static constexpr int kNumJoints = 7;
  using Vector7d = Eigen::Matrix<double, 7, 1>;

  struct Trajectory {
    std::vector<double> times;
    std::vector<Eigen::Vector3d> positions;
    std::vector<Eigen::Quaterniond> orientations;
    std::vector<Eigen::Vector3d> linear_velocities;
  };

  void modeCallback(const std_msgs::msg::String::SharedPtr msg);
  void trajectoryCallback(const trajectory_msgs::msg::MultiDOFJointTrajectory::SharedPtr msg);
  void consumePendingTrajectoryIfReady(Phase phase);
  void updateJointStates();
  void updateCurrentPose();
  void publishStarted();
  void publishFinished();

  // Compute desired Cartesian pose for the current update tick (sets
  // desired_position_/desired_orientation_, returns false if no command should
  // be emitted this tick — e.g., when the sanity guard aborts pre-roll).
  bool computeDesiredPose(const rclcpp::Duration& period);

  // Compute Cartesian impedance joint torques given current state and the
  // desired pose stored in desired_position_/desired_orientation_.
  Vector7d computeImpedanceTorque();

  // Parameters
  std::string arm_id_;
  std::string mode_topic_;
  std::string trajectory_topic_;
  std::string start_topic_;
  std::string finish_topic_;
  bool gazebo_{false};
  bool move_to_start_enabled_{true};
  double move_to_start_min_duration_{4.0};
  double move_to_start_max_translation_velocity_{0.05};
  double move_to_start_max_rotation_velocity_{0.5};

  double translational_stiffness_{200.0};
  double rotational_stiffness_{20.0};
  double nullspace_stiffness_{10.0};
  double joint_damping_ratio_{1.0};

  // Stiffness/damping matrices derived from the scalar parameters
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  Vector7d nullspace_q_target_;

  // Robot model semantic component for FK + Jacobian + Coriolis
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  const std::string k_robot_model_interface_name_{"robot_model"};
  const std::string k_robot_state_interface_name_{"robot_state"};

  // Joint state (read each tick from state_interfaces_)
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;

  // Current EE pose (read each tick via robot_model FK)
  Eigen::Quaterniond current_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d current_position_{Eigen::Vector3d::Zero()};

  // Desired pose (set by phase machinery each tick)
  Eigen::Quaterniond desired_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d desired_position_{Eigen::Vector3d::Zero()};

  // Hold target (captured at HOLD entry)
  Eigen::Quaterniond hold_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d hold_position_{Eigen::Vector3d::Zero()};
  bool hold_initialized_{false};
  std::atomic<bool> hold_reset_requested_{false};

  // Last commanded (smoothed) pose, used as PRE_ROLL start to guarantee
  // continuity with the HOLD reference stream.
  Eigen::Quaterniond last_command_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d last_command_position_{Eigen::Vector3d::Zero()};
  bool last_command_initialized_{false};

  std::atomic<int> phase_{static_cast<int>(Phase::HOLD)};
  std::mutex trajectory_mutex_;
  std::shared_ptr<Trajectory> next_trajectory_;
  std::atomic<bool> has_new_trajectory_{false};
  std::shared_ptr<Trajectory> active_trajectory_;

  bool start_pending_{false};
  bool finish_pending_{false};
  bool tracking_initialized_{false};
  double replay_elapsed_{0.0};
  size_t segment_index_{0};

  Eigen::Quaterniond pre_roll_orientation_start_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d pre_roll_position_start_{Eigen::Vector3d::Zero()};
  double pre_roll_duration_{0.0};
  double pre_roll_elapsed_{0.0};
  int last_pre_roll_log_slot_{-1};

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mode_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr trajectory_sub_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Bool>> start_pub_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Bool>> finish_pub_;
};

}  // namespace franka_example_controllers
