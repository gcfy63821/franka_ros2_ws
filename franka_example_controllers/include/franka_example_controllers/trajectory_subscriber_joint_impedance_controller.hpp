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

#include <string>
#include <vector>
#include <mutex>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

/**
 * The trajectory subscriber joint impedance controller subscribes to joint trajectory
 * messages and follows them using joint impedance control with minimum jerk planning.
 */
class TrajectorySubscriberJointImpedanceController : public controller_interface::ControllerInterface {
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
  bool is_gazebo{false};
  const int num_joints = 7;
  
  // Joint states
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d initial_q_;
  
  // Control gains
  Vector7d k_gains_;
  Vector7d d_gains_;
  
  // Trajectory subscription
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_subscriber_;
  std::string trajectory_topic_{"arm_joint_trajectory"};
  
  // Trajectory data (thread-safe)
  std::mutex trajectory_mutex_;
  Vector7d latest_target_q_;
  bool trajectory_received_{false};
  rclcpp::Time last_trajectory_time_;
  
  // Trajectory following state
  bool reached_first_waypoint_{false};
  double elapsed_time_{0.0};
  double approach_start_time_{0.0};
  double approach_duration_{3.0};  // seconds to reach first waypoint
  double trajectory_timeout_{1.0};  // seconds before considering trajectory stale
  
  // Safety parameters
  double max_joint_distance_{0.5};  // Maximum allowed joint distance (rad) before using smooth approach
  double safe_approach_duration_{4.0};  // Duration for smooth approach when distance threshold exceeded
  bool in_safe_approach_mode_{false};
  Vector7d safe_approach_start_q_;
  double safe_approach_start_time_{0.0};
  bool new_far_target_received_{false};  // Flag to indicate a new far target arrived (thread-safe signal)
  
  // Minimum jerk trajectory planning
  Vector7d computeMinimumJerkTrajectory(const Vector7d& q_start, const Vector7d& q_end,
                                         double t, double duration);
  void trajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void updateJointStates();
  Vector7d getCurrentTrajectoryWaypoint(const rclcpp::Time& current_time);
};

}  // namespace franka_example_controllers

