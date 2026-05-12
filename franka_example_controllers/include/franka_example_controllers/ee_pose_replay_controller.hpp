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
#include <franka_semantic_components/franka_cartesian_pose_interface.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/jntarray.hpp>
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
// Architecture:
//   * EE pose state  : read from franka_cartesian_pose state interface (O_T_EE
//                      broadcast by libfranka). franka_robot_model is NOT used
//                      because its Jacobian/FK accessors return garbage on this
//                      hardware stack.
//   * Jacobian       : computed via KDL from the URDF (pulled from
//                      robot_state_publisher's robot_description parameter).
//   * Torque output  : joint efforts (effort command interface), which bypasses
//                      libfranka's Cartesian motion generator and its
//                      acceleration-discontinuity reflexes.
//
// External protocol is unchanged: mode topic ("hold" / "replay"),
// MultiDOFJointTrajectory on trajectory topic, replay_started / replay_finished
// booleans. Phase machine still goes HOLD -> PRE_ROLL (min-jerk move-to-start)
// -> TRACKING (cubic Hermite over the recorded poses).
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
  using Matrix6x7d = Eigen::Matrix<double, 6, 7>;

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
  bool computeDesiredPose(const rclcpp::Duration& period);
  Vector7d computeImpedanceTorque(const Matrix6x7d& jacobian);

  // Fetch URDF from /robot_state_publisher, build KDL chain + solver.
  bool setupKdlFromUrdf();
  // Compute base-frame Jacobian of the EE for the current q_ using KDL.
  Matrix6x7d computeJacobian() const;

  // Parameters
  std::string arm_id_;
  std::string base_link_;
  std::string ee_link_;
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

  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  Vector7d nullspace_q_target_;

  // EE pose reader (state interface only).
  std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface>
      franka_cartesian_pose_;
  const bool k_elbow_activated_{false};

  // KDL chain + Jacobian solver, built from the URDF in on_configure.
  std::string robot_description_;
  KDL::Chain kdl_chain_;
  std::unique_ptr<KDL::ChainJntToJacSolver> jac_solver_;
  mutable KDL::JntArray kdl_q_;      // reused per tick
  mutable KDL::Jacobian kdl_jac_;    // reused per tick

  // Joint state (read each tick from state_interfaces_)
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;

  // Current EE pose (read each tick via cartesian_pose state interface)
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

  // Captured on first update() tick after activation, when state interfaces
  // hold real values (on_activate runs before read()).
  bool needs_initialization_{true};

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
