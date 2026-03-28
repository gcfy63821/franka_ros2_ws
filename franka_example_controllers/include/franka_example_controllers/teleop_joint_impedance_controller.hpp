// Copyright (c) 2024 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <string>
#include <mutex>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_msgs/action/move.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

/**
 * Teleoperation joint impedance controller for Franka FR3.
 *
 * Receives streaming joint position targets (from external IK solver, e.g. XRoboToolkit/Placo)
 * and tracks them using joint impedance control: tau = k*(q_target - q) + d*(-dq).
 * Gripper is controlled via a separate topic with async action clients.
 *
 * Topics:
 *   - Subscribe: /teleop_joint_commands (sensor_msgs/JointState) — 7 target joint positions
 *   - Subscribe: /teleop_gripper_command (std_msgs/Float64) — gripper width [0, 0.04] m
 *
 * On activation, the controller holds the current position until the first command arrives.
 */
class TeleopJointImpedanceController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;

  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  std::string arm_id_;
  std::string robot_description_;
  const int num_joints_ = 7;

  // Joint state
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d q_target_;
  Vector7d initial_q_;

  // Control gains
  Vector7d k_gains_;
  Vector7d d_gains_;

  // Joint command input
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_command_sub_;
  std::string joint_command_topic_{"/teleop_joint_commands"};
  std::mutex command_mutex_;
  bool command_received_{false};
  double command_timeout_{0.5};
  double max_joint_distance_{0.5};
  double elapsed_time_{0.0};
  double last_command_elapsed_{0.0};
  bool timeout_warned_{false};

  // Gripper command
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr gripper_command_sub_;
  std::string gripper_command_topic_{"/teleop_gripper_command"};
  std::atomic<bool> gripper_is_open_{true};
  std::atomic<bool> gripper_action_pending_{false};
  double gripper_threshold_{0.02};

  // Gripper action clients
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Grasp>> gripper_grasp_client_;
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Move>> gripper_move_client_;
  std::string namespace_;

  // Gazebo mode (skip collision behavior, gripper action clients)
  bool is_gazebo_{false};

  // Helper methods
  void updateJointStates();
  void jointCommandCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void gripperCommandCallback(const std_msgs::msg::Float64::SharedPtr msg);
  void openGripper();
  void closeGripper();
};

}  // namespace franka_example_controllers
