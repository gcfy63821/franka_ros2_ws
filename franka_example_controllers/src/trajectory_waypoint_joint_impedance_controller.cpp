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

#include <franka_example_controllers/trajectory_waypoint_joint_impedance_controller.hpp>
#include <franka_example_controllers/robot_utils.hpp>
#include <franka_example_controllers/default_robot_behavior_utils.hpp>
#include "fmt/format.h"

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
TrajectoryWaypointJointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
TrajectoryWaypointJointImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

TrajectoryWaypointJointImpedanceController::Vector7d 
TrajectoryWaypointJointImpedanceController::computeMinimumJerkTrajectory(
    const TrajectoryWaypointJointImpedanceController::Vector7d& q_start, 
    const TrajectoryWaypointJointImpedanceController::Vector7d& q_end, 
    double t, double duration) {
  if (duration <= 0.0) {
    return q_end;
  }
  
  // Clamp time to [0, duration] to ensure we reach the target exactly at duration
  double t_clamped = std::max(0.0, std::min(t, duration));
  
  // Normalize time to [0, 1]
  double s = t_clamped / duration;
  
  // Minimum jerk polynomial: 10s^3 - 15s^4 + 6s^5
  // This provides smooth acceleration and deceleration with zero velocity at start and end
  // Velocity profile: starts at 0, increases smoothly, peaks in middle, decreases smoothly, ends at 0
  double s2 = s * s;
  double s3 = s2 * s;
  double s4 = s3 * s;
  double s5 = s4 * s;
  
  // Standard minimum jerk polynomial
  // At s=0: poly=0, velocity=0, acceleration=0
  // At s=1: poly=1, velocity=0, acceleration=0
  // Velocity peaks around s=0.5
  double poly = 10.0 * s3 - 15.0 * s4 + 6.0 * s5;
  
  // If we've exceeded the duration, just return the target
  if (t >= duration) {
    return q_end;
  }
  
  return q_start + poly * (q_end - q_start);
}

void TrajectoryWaypointJointImpedanceController::trajectoryCallback(
    const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
  static int callback_count = 0;
  static rclcpp::Time first_callback_time;
  callback_count++;
  
  if (callback_count == 1) {
    first_callback_time = get_node()->get_clock()->now();
    RCLCPP_WARN(get_node()->get_logger(), 
                "*** FIRST TRAJECTORY MESSAGE RECEIVED! ***");
    RCLCPP_WARN(get_node()->get_logger(), 
                "Subscription is working! Topic: '%s'", trajectory_topic_.c_str());
  }
  
  RCLCPP_INFO(get_node()->get_logger(), "=== TRAJECTORY CALLBACK #%d ===", callback_count);
  RCLCPP_INFO(get_node()->get_logger(), "Received trajectory message with %zu points", msg->points.size());
  RCLCPP_INFO(get_node()->get_logger(), "Message header frame_id: '%s'", msg->header.frame_id.c_str());
  RCLCPP_INFO(get_node()->get_logger(), "Message timestamp: sec=%d, nanosec=%u", 
              msg->header.stamp.sec, msg->header.stamp.nanosec);
  
  if (msg->points.empty()) {
    RCLCPP_WARN(get_node()->get_logger(), "Received empty trajectory message");
    return;
  }
  
  // Extract all waypoints from the trajectory
  // Note: trajectory may have 8 positions (7 joints + gripper state)
  std::vector<Vector7d> new_trajectory;
  std::vector<bool> new_gripper_states;
  
  for (size_t i = 0; i < msg->points.size(); ++i) {
    const auto& point = msg->points[i];
    
    // Check if we have at least 7 joints (may have 8 with gripper state)
    if (point.positions.size() < static_cast<size_t>(num_joints)) {
      RCLCPP_WARN(get_node()->get_logger(), 
                  "Trajectory point %zu has %zu joints, expected at least %d. Skipping point.",
                  i, point.positions.size(), num_joints);
      continue;
    }
    
    Vector7d waypoint;
    for (int j = 0; j < num_joints; ++j) {
      waypoint(j) = point.positions[j];
    }
    new_trajectory.push_back(waypoint);
    
    // Extract gripper state (8th position if available)
    // 0.0 = closed, 1.0 = open, default to closed if not specified
    bool gripper_open = false;
    if (point.positions.size() >= static_cast<size_t>(num_joints + 1)) {
      double gripper_value = point.positions[num_joints];
      gripper_open = (gripper_value > 0.5);  // > 0.5 means open
    }
    new_gripper_states.push_back(gripper_open);
  }
  
  if (new_trajectory.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No valid waypoints in trajectory message");
    return;
  }
  
  // Store trajectory (thread-safe)
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    trajectory_ = new_trajectory;
    gripper_states_ = new_gripper_states;
    trajectory_received_ = true;
    trajectory_completed_ = false;
    trajectory_received_time_ = get_node()->get_clock()->now();
  }
  
  // Log gripper states
  std::string gripper_states_str;
  for (size_t i = 0; i < new_gripper_states.size(); ++i) {
    if (i > 0) gripper_states_str += ", ";
    gripper_states_str += (new_gripper_states[i] ? "OPEN" : "CLOSED");
  }
  RCLCPP_INFO(get_node()->get_logger(), 
              "Trajectory gripper states: [%s]", gripper_states_str.c_str());
  
  // Reset waypoint following state
  // Note: elapsed_time_ might not be initialized yet if called during activation
  // Use 0.0 as default, it will be reset properly when controller starts
  current_waypoint_index_ = 0;
  reached_current_waypoint_ = false;
  first_waypoint_reached_ = false;
  current_gripper_state_ = false;  // Reset gripper state
  waypoint_start_time_ = 0.0;  // Will be set properly when controller activates
  // waypoint_start_q_ will be set when we start moving to first waypoint
  
  RCLCPP_INFO(get_node()->get_logger(), 
              "Stored trajectory with %zu waypoints. Starting from waypoint 0.",
              new_trajectory.size());
  RCLCPP_INFO(get_node()->get_logger(),
              "First waypoint: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
              new_trajectory[0](0), new_trajectory[0](1), new_trajectory[0](2),
              new_trajectory[0](3), new_trajectory[0](4), new_trajectory[0](5), new_trajectory[0](6));
  
  // Start signal will be sent when first waypoint is reached (in getCurrentWaypointTarget)
}

TrajectoryWaypointJointImpedanceController::Vector7d 
TrajectoryWaypointJointImpedanceController::getCurrentWaypointTarget() {
  // Read trajectory data (with mutex)
  bool has_trajectory = false;
  size_t num_waypoints = 0;
  size_t current_idx = 0;
  bool completed = false;
  
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    has_trajectory = trajectory_received_ && !trajectory_.empty();
    num_waypoints = trajectory_.size();
    current_idx = current_waypoint_index_;
    completed = trajectory_completed_;
  }
  
  if (!has_trajectory) {
    return q_;  // Return current position if no trajectory
  }
  
  if (completed) {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    return trajectory_.back();  // Hold last waypoint
  }
  
  // Get current waypoint target (read with mutex)
  Vector7d target_waypoint;
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    if (current_idx >= trajectory_.size()) {
      trajectory_completed_ = true;
      RCLCPP_INFO(get_node()->get_logger(), 
                  "Trajectory completed! Reached all %zu waypoints.",
                  trajectory_.size());
      return trajectory_.back();
    }
    target_waypoint = trajectory_[current_idx];
  }
  
  // Initialize waypoint_start_time_ and waypoint_start_q_ if not already set
  // This ensures we start timing and capture start position when we first try to move to a waypoint
  if (waypoint_start_time_ == 0.0 && !reached_current_waypoint_) {
    waypoint_start_time_ = elapsed_time_;
    waypoint_start_q_ = q_;  // Capture current position as start for this waypoint
    
    // Get gripper state for current waypoint and control gripper if needed
    bool target_gripper_state;
    {
      std::lock_guard<std::mutex> lock(trajectory_mutex_);
      if (current_idx < gripper_states_.size()) {
        target_gripper_state = gripper_states_[current_idx];
      } else {
        target_gripper_state = false;  // Default to closed
      }
    }
    
    // Control gripper for current waypoint (especially important for first waypoint)
    if (target_gripper_state != current_gripper_state_) {
      RCLCPP_INFO(get_node()->get_logger(),
                  "Starting waypoint %zu: Changing gripper state: %s -> %s",
                  current_idx + 1,
                  current_gripper_state_ ? "OPEN" : "CLOSED",
                  target_gripper_state ? "OPEN" : "CLOSED");
      controlGripper(target_gripper_state);
      current_gripper_state_ = target_gripper_state;
    } else {
      RCLCPP_DEBUG(get_node()->get_logger(),
                   "Starting waypoint %zu: Gripper state unchanged (%s)",
                   current_idx + 1,
                   current_gripper_state_ ? "OPEN" : "CLOSED");
    }
    
    if (current_idx == 0) {
      // For first waypoint, use initial position instead
      waypoint_start_q_ = initial_q_;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Starting movement to first waypoint. Start: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f], "
                  "Target: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f], Gripper: %s",
                  waypoint_start_q_(0), waypoint_start_q_(1), waypoint_start_q_(2),
                  waypoint_start_q_(3), waypoint_start_q_(4), waypoint_start_q_(5), waypoint_start_q_(6),
                  target_waypoint(0), target_waypoint(1), target_waypoint(2),
                  target_waypoint(3), target_waypoint(4), target_waypoint(5), target_waypoint(6),
                  target_gripper_state ? "OPEN" : "CLOSED");
    } else {
      RCLCPP_INFO(get_node()->get_logger(),
                  "Starting movement to waypoint %zu. Start: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f], "
                  "Target: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f], Gripper: %s",
                  current_idx + 1,
                  waypoint_start_q_(0), waypoint_start_q_(1), waypoint_start_q_(2),
                  waypoint_start_q_(3), waypoint_start_q_(4), waypoint_start_q_(5), waypoint_start_q_(6),
                  target_waypoint(0), target_waypoint(1), target_waypoint(2),
                  target_waypoint(3), target_waypoint(4), target_waypoint(5), target_waypoint(6),
                  target_gripper_state ? "OPEN" : "CLOSED");
    }
  }
  
  // Calculate distance to current waypoint (outside mutex)
  Vector7d error = target_waypoint - q_;
  double max_error = error.cwiseAbs().maxCoeff();
  
  // Check if we've reached the current waypoint
  // Similar to trajectory_following: check both time elapsed and error threshold
  double t_elapsed_check = elapsed_time_ - waypoint_start_time_;
  bool time_reached = (t_elapsed_check >= waypoint_duration_);
  bool error_reached = (max_error < waypoint_tolerance_);
  
  // If time has elapsed, we should have reached the target via minimum jerk
  // If error is small enough, we've reached it via impedance control
  // Use OR logic: reach if either condition is met
  if (!reached_current_waypoint_ && (time_reached || error_reached)) {
    reached_current_waypoint_ = true;
    RCLCPP_INFO(get_node()->get_logger(),
                "Reached waypoint %zu/%zu. Error: %.4f rad, Time: %.2f/%.2f s (time_reached=%d, error_reached=%d)",
                current_idx + 1, num_waypoints, max_error, t_elapsed_check, waypoint_duration_,
                time_reached, error_reached);
    
    // Send start signal when first waypoint is reached
    // This signals the recorder to start continuous recording
    if (current_idx == 0 && !first_waypoint_reached_) {
      first_waypoint_reached_ = true;
      if (start_publisher_) {
        std_msgs::msg::Bool start_msg;
        start_msg.data = true;
        start_publisher_->publish(start_msg);
        RCLCPP_INFO(get_node()->get_logger(), 
                    "Published start signal to topic: %s (reached first waypoint)",
                    start_topic_.c_str());
      }
      RCLCPP_INFO(get_node()->get_logger(), 
                  "First waypoint reached, continuing to next waypoints...");
    }
  }
  
  // If we've reached the current waypoint, prepare to move to next one
  if (reached_current_waypoint_) {
    if (current_idx < num_waypoints - 1) {
      // Move to next waypoint
      {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        current_waypoint_index_ = current_idx + 1;
      }
      reached_current_waypoint_ = false;
      waypoint_start_time_ = elapsed_time_;  // Reset timer for new waypoint
      waypoint_start_q_ = q_;  // Capture current position as start for next waypoint
      
      // Get next waypoint target and gripper state
      bool target_gripper_state;
      {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        target_waypoint = trajectory_[current_waypoint_index_];
        if (current_waypoint_index_ < gripper_states_.size()) {
          target_gripper_state = gripper_states_[current_waypoint_index_];
        } else {
          target_gripper_state = false;  // Default to closed
        }
      }
      
      // Control gripper if state changed (do this before starting to move)
      if (target_gripper_state != current_gripper_state_) {
        RCLCPP_INFO(get_node()->get_logger(),
                    "Changing gripper state: %s -> %s",
                    current_gripper_state_ ? "OPEN" : "CLOSED",
                    target_gripper_state ? "OPEN" : "CLOSED");
        controlGripper(target_gripper_state);
        current_gripper_state_ = target_gripper_state;
      }
      
      RCLCPP_INFO(get_node()->get_logger(),
                  "Starting movement to waypoint %zu/%zu. Start: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f], "
                  "Target: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f] (gripper: %s)",
                  current_waypoint_index_ + 1, num_waypoints,
                  waypoint_start_q_(0), waypoint_start_q_(1), waypoint_start_q_(2),
                  waypoint_start_q_(3), waypoint_start_q_(4), waypoint_start_q_(5), waypoint_start_q_(6),
                  target_waypoint(0), target_waypoint(1), target_waypoint(2),
                  target_waypoint(3), target_waypoint(4), target_waypoint(5), target_waypoint(6),
                  target_gripper_state ? "OPEN" : "CLOSED");
      
      // Continue to use minimum jerk from captured start position to new target
      // (will be handled in the code below)
    } else {
      // All waypoints reached
      {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        trajectory_completed_ = true;
      }
      RCLCPP_INFO(get_node()->get_logger(), 
                  "Trajectory completed! Reached all %zu waypoints.",
                  num_waypoints);
      std::lock_guard<std::mutex> lock(trajectory_mutex_);
      return trajectory_.back();
    }
  }
  
  // Use minimum jerk to move to current waypoint (for ALL waypoints)
  // Use the captured start position (waypoint_start_q_) which was set when we started this waypoint
  // This ensures consistent trajectory from a fixed start point, not a dynamically changing current position
  double t_elapsed = elapsed_time_ - waypoint_start_time_;
  
  // Clamp elapsed time to duration to prevent overshooting
  // The minimum jerk function will handle clamping, but we ensure smooth behavior here
  if (t_elapsed >= waypoint_duration_) {
    // If we've exceeded duration, just return target (should have been caught above, but safety check)
    return target_waypoint;
  }
  
  // Use minimum jerk trajectory for ALL waypoints
  // This provides smooth acceleration and deceleration with zero velocity at start and end
  // The trajectory goes from waypoint_start_q_ (captured when starting) to target_waypoint
  Vector7d q_goal = computeMinimumJerkTrajectory(waypoint_start_q_, target_waypoint, t_elapsed, waypoint_duration_);
  
  return q_goal;
}

void TrajectoryWaypointJointImpedanceController::publishFinishSignal() {
  if (finish_publisher_ && trajectory_completed_) {
    std_msgs::msg::Bool finish_msg;
    finish_msg.data = true;
    finish_publisher_->publish(finish_msg);
    RCLCPP_INFO(get_node()->get_logger(), "Published finish signal to topic: %s", finish_topic_.c_str());
  }
}

controller_interface::return_type TrajectoryWaypointJointImpedanceController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  updateJointStates();
  elapsed_time_ += period.seconds();
  
  Vector7d q_goal = getCurrentWaypointTarget();
  
  // Check if trajectory is complete and publish finish signal (only once)
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    static bool finish_signal_published = false;
    if (trajectory_completed_ && !finish_signal_published) {
      publishFinishSignal();
      finish_signal_published = true;
    } else if (!trajectory_completed_) {
      finish_signal_published = false;  // Reset when new trajectory arrives
    }
  }
  
  // Velocity filtering
  const double kAlpha = 0.99;
  dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;
  
  // Joint impedance control law
  Vector7d error = q_goal - q_;
  Vector7d tau_d_calculated =
      k_gains_.cwiseProduct(error) + d_gains_.cwiseProduct(-dq_filtered_);
  
  // Debug logging (throttled)
  static double last_debug_time = 0.0;
  if (elapsed_time_ - last_debug_time >= 2.0) {
    double max_error = error.cwiseAbs().maxCoeff();
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    RCLCPP_DEBUG(get_node()->get_logger(),
                 "Control update: max_error=%.4f, waypoint=%zu/%zu, completed=%d",
                 max_error, current_waypoint_index_ + 1, 
                 trajectory_.empty() ? 0 : trajectory_.size(), trajectory_completed_);
    last_debug_time = elapsed_time_;
  }
  
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }
  
  return controller_interface::return_type::OK;
}

CallbackReturn TrajectoryWaypointJointImpedanceController::on_init() {
  try {
    auto_declare<bool>("gazebo", false);
    auto_declare<std::string>("arm_id", "fr3");  // Default to "fr3"
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<std::string>("trajectory_topic", "arm_joint_trajectory");
    auto_declare<std::string>("finish_topic", "trajectory_finished");
    auto_declare<std::string>("start_topic", "recording_start");
    auto_declare<double>("waypoint_duration", 2.0);
    auto_declare<double>("waypoint_tolerance", 0.05);
    
    // Read arm_id early so it's available for command_interface_configuration()
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    if (arm_id_.empty()) {
      arm_id_ = "fr3";  // Fallback default
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn TrajectoryWaypointJointImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  is_gazebo = get_node()->get_parameter("gazebo").as_bool();
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  trajectory_topic_ = get_node()->get_parameter("trajectory_topic").as_string();
  finish_topic_ = get_node()->get_parameter("finish_topic").as_string();
  start_topic_ = get_node()->get_parameter("start_topic").as_string();
  waypoint_duration_ = get_node()->get_parameter("waypoint_duration").as_double();
  waypoint_tolerance_ = get_node()->get_parameter("waypoint_tolerance").as_double();
  
  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  
  if (k_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains parameter not set");
    return CallbackReturn::FAILURE;
  }
  if (k_gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains should be of size %d but is of size %ld",
                 num_joints, k_gains.size());
    return CallbackReturn::FAILURE;
  }
  if (d_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains parameter not set");
    return CallbackReturn::FAILURE;
  }
  if (d_gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains should be of size %d but is of size %ld",
                 num_joints, d_gains.size());
    return CallbackReturn::FAILURE;
  }
  
  for (int i = 0; i < num_joints; ++i) {
    d_gains_(i) = d_gains.at(i);
    k_gains_(i) = k_gains.at(i);
  }
  dq_filtered_.setZero();
  
  // Get robot description (similar to trajectory_subscriber)
  RCLCPP_INFO(get_node()->get_logger(), "Getting robot_description parameter...");
  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  
  if (!parameters_client->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_WARN(get_node()->get_logger(), "robot_state_publisher service not available, using arm_id parameter");
  } else {
    auto future = parameters_client->get_parameters({"robot_description"});
    auto result = future.get();
    if (!result.empty()) {
      robot_description_ = result[0].value_to_string();
      RCLCPP_INFO(get_node()->get_logger(), "Got robot_description parameter");
    } else {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
    }
    
    arm_id_ = robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());
    RCLCPP_INFO(get_node()->get_logger(), "Extracted arm_id from robot_description: '%s'", arm_id_.c_str());
  }
  
  // If arm_id is still empty, use the parameter directly
  if (arm_id_.empty()) {
    RCLCPP_WARN(get_node()->get_logger(), "arm_id is empty, using parameter value");
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
  }
  
  RCLCPP_INFO(get_node()->get_logger(), "Final arm_id: '%s'", arm_id_.c_str());
  
  // Get namespace for gripper actions
  namespace_ = get_node()->get_namespace();
  
  // Create gripper action clients (only for real hardware)
  // Use absolute paths (starting with '/') to avoid namespace issues
  // The gripper node typically runs in root namespace, not in controller namespace
  if (!is_gazebo) {
    std::string grasp_action = fmt::format("{}/franka_gripper/grasp", namespace_);
    std::string move_action = fmt::format("{}/franka_gripper/move", namespace_);

    gripper_grasp_action_client_ = rclcpp_action::create_client<franka_msgs::action::Grasp>(
        get_node(), grasp_action);

    gripper_move_action_client_ = rclcpp_action::create_client<franka_msgs::action::Move>(
        get_node(), move_action);

    RCLCPP_INFO(get_node()->get_logger(),
                "Gripper action clients created (namespace: '%s')", namespace_.c_str());
    RCLCPP_INFO(get_node()->get_logger(),
                "Gripper Grasp action topic: %s", grasp_action.c_str());
    RCLCPP_INFO(get_node()->get_logger(),
                "Gripper Move action topic: %s", move_action.c_str());
  } else {
    RCLCPP_INFO(get_node()->get_logger(), 
                "Gazebo mode: Gripper action clients not created (gripper control disabled)");
  }
  
  // Load collision behavior (only for real hardware)
  if (!is_gazebo) {
    auto client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
        "service_server/set_full_collision_behavior");
    auto request = DefaultRobotBehavior::getDefaultCollisionBehaviorRequest();
    
    auto future_result = client->async_send_request(request);
    future_result.wait_for(robot_utils::time_out);
    
    auto success = future_result.get();
    if (!success) {
      RCLCPP_FATAL(get_node()->get_logger(), "Failed to set default collision behavior.");
      return CallbackReturn::ERROR;
    } else {
      RCLCPP_INFO(get_node()->get_logger(), "Default collision behavior set.");
    }
  }
  
  // Create subscription with absolute topic name (similar to trajectory_subscriber)
  std::string absolute_topic = trajectory_topic_;
  if (!absolute_topic.empty() && absolute_topic[0] != '/') {
    absolute_topic = "/" + absolute_topic;
  }
  
  std::string node_namespace = std::string(get_node()->get_namespace());
  RCLCPP_INFO(get_node()->get_logger(), "Node namespace: '%s'", node_namespace.c_str());
  RCLCPP_INFO(get_node()->get_logger(), "Creating subscription to topic: '%s' (absolute: '%s')", 
              trajectory_topic_.c_str(), absolute_topic.c_str());
  RCLCPP_INFO(get_node()->get_logger(), "Subscription type: trajectory_msgs/msg/JointTrajectory");
  RCLCPP_WARN(get_node()->get_logger(), 
              "IMPORTANT: Subscription will listen to absolute topic '%s' (namespace-independent)", 
              absolute_topic.c_str());
  
  // Create subscription with explicit topic name resolution
  std::string resolved_topic = absolute_topic;
  if (resolved_topic[0] != '/') {
    resolved_topic = "/" + resolved_topic;
  }
  
  RCLCPP_INFO(get_node()->get_logger(), 
              "Creating subscription with resolved topic: '%s'", resolved_topic.c_str());
  
  // Use best-effort QoS to match typical publisher settings
  rclcpp::QoS qos_profile = rclcpp::SystemDefaultsQoS();
  qos_profile.reliability(rclcpp::ReliabilityPolicy::BestEffort);
  qos_profile.durability(rclcpp::DurabilityPolicy::Volatile);
  
  RCLCPP_INFO(get_node()->get_logger(), 
              "QoS: reliability=BestEffort, durability=Volatile");
  
  trajectory_subscriber_ = get_node()->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      resolved_topic,
      qos_profile,
      std::bind(&TrajectoryWaypointJointImpedanceController::trajectoryCallback,
                this, std::placeholders::_1));
  
  RCLCPP_WARN(get_node()->get_logger(), 
              "Subscription created with topic: '%s' (should work across all namespaces)", 
              resolved_topic.c_str());
  
  // Create publishers in on_configure (they can be used in on_activate)
  std::string resolved_finish_topic = finish_topic_;
  if (!resolved_finish_topic.empty() && resolved_finish_topic[0] != '/') {
    resolved_finish_topic = "/" + resolved_finish_topic;
  }
  finish_publisher_ = get_node()->create_publisher<std_msgs::msg::Bool>(
      resolved_finish_topic, 10);
  
  std::string resolved_start_topic = start_topic_;
  if (!resolved_start_topic.empty() && resolved_start_topic[0] != '/') {
    resolved_start_topic = "/" + resolved_start_topic;
  }
  start_publisher_ = get_node()->create_publisher<std_msgs::msg::Bool>(
      resolved_start_topic, 10);
  
  RCLCPP_INFO(get_node()->get_logger(), 
              "Trajectory subscription created. Waiting for trajectory messages...");
  RCLCPP_INFO(get_node()->get_logger(),
              "Configuration: waypoint_duration=%.2f s, waypoint_tolerance=%.4f rad",
              waypoint_duration_, waypoint_tolerance_);
  RCLCPP_INFO(get_node()->get_logger(),
              "Publishers created: finish='%s', start='%s'",
              resolved_finish_topic.c_str(), resolved_start_topic.c_str());
  
  return CallbackReturn::SUCCESS;
}

CallbackReturn TrajectoryWaypointJointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  // Verify command interfaces are available
  if (command_interfaces_.size() != static_cast<size_t>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "Expected %d command interfaces, got %zu",
                 num_joints, command_interfaces_.size());
    return CallbackReturn::ERROR;
  }
  
  // Verify state interfaces are available
  if (state_interfaces_.size() != static_cast<size_t>(2 * num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "Expected %d state interfaces, got %zu",
                 2 * num_joints, state_interfaces_.size());
    return CallbackReturn::ERROR;
  }
  
  updateJointStates();
  dq_filtered_.setZero();
  initial_q_ = q_;
  waypoint_start_q_ = q_;  // Initialize start position
  elapsed_time_ = 0.0;
  waypoint_start_time_ = 0.0;
  current_waypoint_index_ = 0;
  reached_current_waypoint_ = false;
  first_waypoint_reached_ = false;
  current_gripper_state_ = false;  // Start with gripper closed
  
  // If trajectory was already received before activation, prepare for execution
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    if (trajectory_received_ && !trajectory_.empty()) {
      // Reset waypoint following state
      current_waypoint_index_ = 0;
      reached_current_waypoint_ = false;
      first_waypoint_reached_ = false;
      waypoint_start_time_ = 0.0;  // Will be set when we start moving to first waypoint
      current_gripper_state_ = false;  // Reset gripper state
      
      // Set initial gripper state for first waypoint if available
      if (!gripper_states_.empty()) {
        current_gripper_state_ = gripper_states_[0];
      }
      
      RCLCPP_INFO(get_node()->get_logger(),
                  "Trajectory already received with %zu waypoints. Ready to execute.",
                  trajectory_.size());
    }
  }
  
  // Wait for gripper action servers in on_activate (OK to block here, not in control loop)
  // Only for real hardware (gripper not available in Gazebo)
  if (!is_gazebo) {
    std::string move_action = fmt::format("{}/franka_gripper/move", namespace_);
    std::string grasp_action = fmt::format("{}/franka_gripper/grasp", namespace_);

    if (gripper_move_action_client_ && !gripper_move_action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_node()->get_logger(), "Gripper Move action server not available after waiting 5 seconds");
      RCLCPP_ERROR(get_node()->get_logger(), "Expected topic: %s", move_action.c_str());
      RCLCPP_ERROR(get_node()->get_logger(), "Check if gripper node is running: ros2 node list | grep gripper");
      RCLCPP_ERROR(get_node()->get_logger(), "Check available actions: ros2 action list | grep gripper");
    } else if (gripper_move_action_client_) {
      RCLCPP_INFO(get_node()->get_logger(), "Gripper Move action server is ready at %s", move_action.c_str());
    }
    if (gripper_grasp_action_client_ && !gripper_grasp_action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_node()->get_logger(), "Gripper Grasp action server not available after waiting 5 seconds");
      RCLCPP_ERROR(get_node()->get_logger(), "Expected topic: %s", grasp_action.c_str());
      RCLCPP_ERROR(get_node()->get_logger(), "Check if gripper node is running: ros2 node list | grep gripper");
      RCLCPP_ERROR(get_node()->get_logger(), "Check available actions: ros2 action list | grep gripper");
    } else if (gripper_grasp_action_client_) {
      RCLCPP_INFO(get_node()->get_logger(), "Gripper Grasp action server is ready at %s", grasp_action.c_str());
    }
  }
  
  // Don't clear trajectory if it was already received - keep it for execution
  // Only reset completion flag
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    trajectory_completed_ = false;
  }
  
  // Publishers should already be created in on_configure, but verify they exist
  std::string resolved_finish_topic = finish_topic_;
  if (!resolved_finish_topic.empty() && resolved_finish_topic[0] != '/') {
    resolved_finish_topic = "/" + resolved_finish_topic;
  }
  if (!finish_publisher_) {
    finish_publisher_ = get_node()->create_publisher<std_msgs::msg::Bool>(
        resolved_finish_topic, 10);
  }
  
  std::string resolved_start_topic = start_topic_;
  if (!resolved_start_topic.empty() && resolved_start_topic[0] != '/') {
    resolved_start_topic = "/" + resolved_start_topic;
  }
  if (!start_publisher_) {
    start_publisher_ = get_node()->create_publisher<std_msgs::msg::Bool>(
        resolved_start_topic, 10);
  }
  
  first_waypoint_reached_ = false;
  
  // Get absolute topic name for logging
  std::string absolute_topic = trajectory_topic_;
  if (!absolute_topic.empty() && absolute_topic[0] != '/') {
    absolute_topic = "/" + absolute_topic;
  }
  
  RCLCPP_INFO(get_node()->get_logger(), 
              "=== TRAJECTORY WAYPOINT CONTROLLER ACTIVATED ===");
  RCLCPP_INFO(get_node()->get_logger(), 
              "Waiting for trajectory messages on topic: '%s'", absolute_topic.c_str());
  RCLCPP_INFO(get_node()->get_logger(), 
              "Node namespace: '%s'", get_node()->get_namespace());
  RCLCPP_INFO(get_node()->get_logger(), 
              "Expected message type: trajectory_msgs/msg/JointTrajectory");
  RCLCPP_WARN(get_node()->get_logger(), 
              "IMPORTANT: Controller subscribes to '%s' (trajectory_msgs/JointTrajectory)", 
              absolute_topic.c_str());
  RCLCPP_WARN(get_node()->get_logger(), 
              "This is DIFFERENT from '/arm_joint_positions' (sensor_msgs/JointState)!");
  RCLCPP_INFO(get_node()->get_logger(), 
              "To verify subscription, run: ros2 topic info %s", absolute_topic.c_str());
  RCLCPP_INFO(get_node()->get_logger(), 
              "To check if topic exists: ros2 topic list | grep %s", absolute_topic.c_str());
  RCLCPP_INFO(get_node()->get_logger(),
              "Finish signal will be published to: '%s'", resolved_finish_topic.c_str());
  RCLCPP_INFO(get_node()->get_logger(),
              "Start signal will be published to: '%s' (when reaching first waypoint)",
              resolved_start_topic.c_str());
  RCLCPP_INFO(get_node()->get_logger(), 
              "Initial position: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
              initial_q_(0), initial_q_(1), initial_q_(2), initial_q_(3),
              initial_q_(4), initial_q_(5), initial_q_(6));
  RCLCPP_INFO(get_node()->get_logger(), 
              "is_gazebo: %s", is_gazebo ? "true" : "false");
  
  return CallbackReturn::SUCCESS;
}

void TrajectoryWaypointJointImpedanceController::updateJointStates() {
  static bool first_call = true;
  
  if (first_call) {
    RCLCPP_INFO(get_node()->get_logger(), "=== FIRST updateJointStates CALL ===");
    RCLCPP_INFO(get_node()->get_logger(), "State interfaces count: %zu", state_interfaces_.size());
    RCLCPP_INFO(get_node()->get_logger(), "Expected: %d (2 per joint: position + velocity)", num_joints * 2);
    
    for (size_t i = 0; i < state_interfaces_.size() && i < 14; ++i) {
      RCLCPP_INFO(get_node()->get_logger(), 
                  "State interface[%zu]: name='%s', value=%.3f",
                  i, state_interfaces_[i].get_interface_name().c_str(), 
                  state_interfaces_[i].get_value());
    }
    first_call = false;
  }
  
  if (state_interfaces_.size() < static_cast<size_t>(num_joints * 2)) {
    RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5.0,
                          "Not enough state interfaces: got %zu, expected %d",
                          state_interfaces_.size(), num_joints * 2);
    return;
  }
  
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);
    
    if (position_interface.get_interface_name() != "position") {
      RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5.0,
                            "Position interface[%d] name mismatch: expected 'position', got '%s'",
                            i, position_interface.get_interface_name().c_str());
    }
    if (velocity_interface.get_interface_name() != "velocity") {
      RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5.0,
                            "Velocity interface[%d] name mismatch: expected 'velocity', got '%s'",
                            i, velocity_interface.get_interface_name().c_str());
    }
    
    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

void TrajectoryWaypointJointImpedanceController::controlGripper(bool open) {
  if (is_gazebo) {
    // Skip gripper control in Gazebo (gripper not available)
    RCLCPP_DEBUG(get_node()->get_logger(), 
                 "Gazebo mode: Skipping gripper control (requested: %s)",
                 open ? "OPEN" : "CLOSED");
    return;
  }
  
  // Check if action clients are available
  if (!gripper_move_action_client_ || !gripper_grasp_action_client_) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5.0,
                         "Gripper action clients not available, skipping gripper control");
    return;
  }
  
  RCLCPP_INFO(get_node()->get_logger(), 
              "Attempting to %s gripper...", open ? "OPEN" : "CLOSE");
  
  if (open) {
    openGripper();
  } else {
    closeGripper();
  }
}

bool TrajectoryWaypointJointImpedanceController::openGripper() {
  if (!gripper_move_action_client_) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5.0,
                         "Gripper Move action client not available");
    return false;
  }
  
  // Non-blocking check: only proceed if action server is ready
  // Do NOT wait here - this is called from the real-time control loop!
  if (!gripper_move_action_client_->action_server_is_ready()) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5.0,
                         "Gripper Move action server not ready (check if gripper node is running)");
    return false;
  }
  
  RCLCPP_INFO(get_node()->get_logger(), "Opening gripper to 0.08m");
  
  franka_msgs::action::Move::Goal move_goal;
  move_goal.width = 0.08;  // Open to 80mm
  move_goal.speed = 0.2;
  
  // Send goal asynchronously - do NOT wait for result in control loop!
  // Use empty goal options (callbacks not needed for fire-and-forget)
  auto goal_handle_future = gripper_move_action_client_->async_send_goal(move_goal);
  
  // Check if future is valid (non-blocking check)
  if (!goal_handle_future.valid()) {
    RCLCPP_WARN(get_node()->get_logger(), "Failed to send gripper Move goal (future invalid)");
    return false;
  }
  
  return true;
}

bool TrajectoryWaypointJointImpedanceController::closeGripper() {
  if (!gripper_grasp_action_client_) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5.0,
                         "Gripper Grasp action client not available");
    return false;
  }
  
  // Non-blocking check: only proceed if action server is ready
  // Do NOT wait here - this is called from the real-time control loop!
  if (!gripper_grasp_action_client_->action_server_is_ready()) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5.0,
                         "Gripper Grasp action server not ready (check if gripper node is running)");
    return false;
  }
  
  RCLCPP_INFO(get_node()->get_logger(), "Closing gripper (target width: 0.015m, force: 100N)");
  
  franka_msgs::action::Grasp::Goal grasp_goal;
  grasp_goal.width = 0.015;  // Target width 15mm
  grasp_goal.speed = 0.05;
  grasp_goal.force = 100.0;
  grasp_goal.epsilon.inner = 0.005;
  grasp_goal.epsilon.outer = 0.010;
  
  // Send goal asynchronously - do NOT wait for result in control loop!
  // Use empty goal options (callbacks not needed for fire-and-forget)
  auto goal_handle_future = gripper_grasp_action_client_->async_send_goal(grasp_goal);
  
  // Check if future is valid (non-blocking check)
  if (!goal_handle_future.valid()) {
    RCLCPP_WARN(get_node()->get_logger(), "Failed to send gripper Grasp goal (future invalid)");
    return false;
  }
  
  return true;
}

}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::TrajectoryWaypointJointImpedanceController,
                       controller_interface::ControllerInterface)

