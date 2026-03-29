// Dexterous hand joint impedance controller for Franka FR3 + Sharpa HA4.
//
// Simplified version of TeleopJointImpedanceController without gripper dependencies.
// Receives streaming joint position targets via ROS2 and tracks them using
// joint impedance control: tau = k*(q_target - q) + d*(-dq).
//
// Topics:
//   - Subscribe: /teleop_joint_commands (sensor_msgs/JointState) — 7 target joint positions
//
// On activation, holds current position until the first command arrives.

#pragma once

#include <string>
#include <mutex>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

class DexhandJointImpedanceController : public controller_interface::ControllerInterface {
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

  // Helper methods
  void updateJointStates();
  void jointCommandCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
};

}  // namespace franka_example_controllers
