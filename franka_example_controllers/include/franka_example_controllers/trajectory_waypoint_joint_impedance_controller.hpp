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
#include <rclcpp_action/rclcpp_action.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <std_msgs/msg/bool.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_msgs/action/move.hpp>
#include "fmt/format.h"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

/**
 * The trajectory waypoint joint impedance controller receives a full joint trajectory
 * from a ROS topic, moves to each waypoint sequentially using minimum jerk trajectories,
 * and publishes a finish signal when the trajectory is complete.
 */
class TrajectoryWaypointJointImpedanceController : public controller_interface::ControllerInterface {
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
  
  // Finish signal publisher
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr finish_publisher_;
  std::string finish_topic_{"trajectory_finished"};
  
  // Start signal publisher (published when reaching first waypoint)
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr start_publisher_;
  std::string start_topic_{"recording_start"};
  bool first_waypoint_reached_{false};
  
  // Trajectory data (thread-safe)
  std::mutex trajectory_mutex_;
  std::vector<Vector7d> trajectory_;
  std::vector<bool> gripper_states_;  // true = open, false = closed
  bool trajectory_received_{false};
  bool trajectory_completed_{false};
  rclcpp::Time trajectory_received_time_;
  
  // Waypoint following state
  size_t current_waypoint_index_{0};
  bool reached_current_waypoint_{false};
  double elapsed_time_{0.0};
  double waypoint_start_time_{0.0};
  Vector7d waypoint_start_q_;  // Start position for current waypoint (captured when transitioning)
  double waypoint_duration_{5.0};  // seconds to reach each waypoint
  double waypoint_tolerance_{0.05};  // rad tolerance to consider waypoint reached
  bool current_gripper_state_{false};  // Current gripper state (true = open, false = closed)
  
  // Gripper action clients
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Grasp>> gripper_grasp_action_client_;
  std::shared_ptr<rclcpp_action::Client<franka_msgs::action::Move>> gripper_move_action_client_;
  std::string namespace_;
  
  // Minimum jerk trajectory planning
  Vector7d computeMinimumJerkTrajectory(const Vector7d& q_start, const Vector7d& q_end,
                                         double t, double duration);
  void trajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  void updateJointStates();
  Vector7d getCurrentWaypointTarget();
  void publishFinishSignal();
  void controlGripper(bool open);
  bool openGripper();
  bool closeGripper();
};

}  // namespace franka_example_controllers

