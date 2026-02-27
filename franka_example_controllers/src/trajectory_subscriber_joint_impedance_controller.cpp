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

#include <franka_example_controllers/trajectory_subscriber_joint_impedance_controller.hpp>
#include <franka_example_controllers/robot_utils.hpp>
#include <franka_example_controllers/default_robot_behavior_utils.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
TrajectorySubscriberJointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
TrajectorySubscriberJointImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

TrajectorySubscriberJointImpedanceController::Vector7d 
TrajectorySubscriberJointImpedanceController::computeMinimumJerkTrajectory(
    const TrajectorySubscriberJointImpedanceController::Vector7d& q_start, 
    const TrajectorySubscriberJointImpedanceController::Vector7d& q_end, 
    double t, double duration) {
  if (duration <= 0.0) {
    return q_end;
  }
  
  // Normalize time to [0, 1]
  double s = std::max(0.0, std::min(1.0, t / duration));
  
  // Minimum jerk polynomial: s^3 * (10 - 15*s + 6*s^2)
  double s3 = s * s * s;
  double s4 = s3 * s;
  double s5 = s4 * s;
  double poly = 10.0 * s3 - 15.0 * s4 + 6.0 * s5;
  
  return q_start + poly * (q_end - q_start);
}

void TrajectorySubscriberJointImpedanceController::trajectoryCallback(
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
  
  // Extract joint positions from the first (and typically only) point
  const auto& point = msg->points[0];
  
  RCLCPP_INFO(get_node()->get_logger(), "Trajectory point has %zu positions", point.positions.size());
  if (!msg->joint_names.empty()) {
    std::string joint_names_str;
    for (size_t i = 0; i < msg->joint_names.size(); ++i) {
      if (i > 0) joint_names_str += ", ";
      joint_names_str += msg->joint_names[i];
    }
    RCLCPP_INFO(get_node()->get_logger(), "Joint names in message: [%s]", joint_names_str.c_str());
  }
  
  if (point.positions.size() != static_cast<size_t>(num_joints)) {
    RCLCPP_ERROR(get_node()->get_logger(), 
                 "Trajectory point has %zu joints, expected %d. Cannot process.",
                 point.positions.size(), num_joints);
    return;
  }
  
  // Verify joint names match (if provided)
  if (!msg->joint_names.empty()) {
    // Check if joint names match expected pattern (fr3_joint1, fr3_joint2, etc.)
    bool names_match = true;
    for (size_t i = 0; i < msg->joint_names.size() && i < static_cast<size_t>(num_joints); ++i) {
      std::string expected_name = arm_id_ + "_joint" + std::to_string(i + 1);
      if (msg->joint_names[i] != expected_name) {
        RCLCPP_WARN(get_node()->get_logger(), 
                    "Joint name mismatch at index %zu: expected '%s', got '%s'",
                    i, expected_name.c_str(), msg->joint_names[i].c_str());
        names_match = false;
      }
    }
    if (!names_match) {
      RCLCPP_WARN(get_node()->get_logger(), 
                  "Joint names in trajectory don't match expected pattern (arm_id='%s'), but continuing",
                  arm_id_.c_str());
    } else {
      RCLCPP_INFO(get_node()->get_logger(), "Joint names match expected pattern");
    }
  }
  
  // Get current position for safety check (read outside mutex, but be careful with thread safety)
  // Note: q_ is updated in update() loop, but reading it here is generally safe for distance check
  Vector7d current_q;
  {
    // We need to read q_ safely - it's updated in update() which runs in controller thread
    // For safety check, we'll read it and use it to determine if we need safe approach
    // The actual safe approach will be handled in the update loop
    current_q = q_;  // This read is not thread-safe, but acceptable for distance estimation
  }
  
  // Store the target joint positions (thread-safe)
  // Always update to the LATEST received position
  Vector7d previous_target;
  Vector7d new_target;
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    previous_target = latest_target_q_;
    
    // Store new target
    for (int i = 0; i < num_joints; ++i) {
      new_target(i) = point.positions[i];
      latest_target_q_(i) = point.positions[i];
    }
    trajectory_received_ = true;
    last_trajectory_time_ = msg->header.stamp;
  }
  
  // Calculate distance from CURRENT position to NEW target (safety check)
  Vector7d distance_to_new_target = new_target - current_q;
  double max_distance_to_new = distance_to_new_target.cwiseAbs().maxCoeff();
  double euclidean_distance_to_new = distance_to_new_target.norm();
  
  // Calculate change from previous target (for logging)
  Vector7d target_change = new_target - previous_target;
  double max_change = target_change.cwiseAbs().maxCoeff();
  
  RCLCPP_INFO(get_node()->get_logger(), 
              "Stored NEW trajectory point: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
              new_target(0), new_target(1), new_target(2),
              new_target(3), new_target(4), new_target(5), new_target(6));
  
  // Safety check: if new target is far from current position, set flag for safe approach
  // The update loop will handle the actual safe approach timing and reset
  if (max_distance_to_new > max_joint_distance_) {
    // Set flag to signal that a new far target arrived
    // The update loop will detect this and reset safe approach mode properly
    new_far_target_received_ = true;
    
    RCLCPP_WARN(get_node()->get_logger(),
                "NEW TARGET IS FAR FROM CURRENT POSITION!");
    RCLCPP_WARN(get_node()->get_logger(),
                "Distance to new target: max_joint_error=%.4f rad (threshold=%.4f rad), "
                "euclidean_distance=%.4f rad",
                max_distance_to_new, max_joint_distance_, euclidean_distance_to_new);
    RCLCPP_INFO(get_node()->get_logger(),
                "Current position: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                current_q(0), current_q(1), current_q(2), current_q(3),
                current_q(4), current_q(5), current_q(6));
    RCLCPP_INFO(get_node()->get_logger(),
                "New target: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                new_target(0), new_target(1), new_target(2), new_target(3),
                new_target(4), new_target(5), new_target(6));
    RCLCPP_WARN(get_node()->get_logger(),
                "Controller will enter SAFE APPROACH MODE in update loop - will move smoothly over %.2f seconds",
                safe_approach_duration_);
  } else {
    // Target is close - clear the flag
    new_far_target_received_ = false;
    if (max_change > 0.001) {
      RCLCPP_INFO(get_node()->get_logger(), 
                  "Target changed by max %.4f rad - distance to current: %.4f rad (within threshold, direct tracking)",
                  max_change, max_distance_to_new);
    } else {
      RCLCPP_DEBUG(get_node()->get_logger(), 
                   "Target unchanged or very close change (%.4f rad)", max_change);
    }
  }
  
  RCLCPP_INFO(get_node()->get_logger(), "Trajectory received flag set to true - controller will track this position");
}

TrajectorySubscriberJointImpedanceController::Vector7d 
TrajectorySubscriberJointImpedanceController::getCurrentTrajectoryWaypoint(
    const rclcpp::Time& current_time) {
  Vector7d target_q;
  bool trajectory_valid = false;
  
  // Get target from trajectory (protected by mutex)
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    
  // Check if we have received any trajectory
  if (!trajectory_received_) {
    static double last_warn_time = 0.0;
    static int warn_count = 0;
    double current_time_sec = current_time.seconds();
    if (current_time_sec - last_warn_time >= 5.0) {
      warn_count++;
      RCLCPP_WARN(get_node()->get_logger(),
                  "No trajectory received yet (elapsed=%.2f s, warn_count=%d), holding current position",
                  elapsed_time_, warn_count);
      
      // Get absolute topic name for logging
      std::string absolute_topic = trajectory_topic_;
      if (!absolute_topic.empty() && absolute_topic[0] != '/') {
        absolute_topic = "/" + absolute_topic;
      }
      
      if (warn_count == 1 || warn_count % 5 == 0) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "TROUBLESHOOTING: Controller is subscribed to: '%s'", absolute_topic.c_str());
        RCLCPP_WARN(get_node()->get_logger(),
                    "Expected message type: trajectory_msgs/msg/JointTrajectory");
        RCLCPP_WARN(get_node()->get_logger(),
                    "Check if publisher exists: ros2 topic list | grep %s", absolute_topic.c_str());
        RCLCPP_WARN(get_node()->get_logger(),
                    "Check topic info: ros2 topic info %s", absolute_topic.c_str());
        RCLCPP_WARN(get_node()->get_logger(),
                    "Note: /arm_joint_positions (sensor_msgs/JointState) is DIFFERENT from %s",
                    absolute_topic.c_str());
      }
      last_warn_time = current_time_sec;
    }
    return q_;  // Return current position (q_ is safe to access here)
  }
    
    // Check if trajectory is stale (timeout)
    double time_since_last_trajectory = (current_time - last_trajectory_time_).seconds();
    if (time_since_last_trajectory > trajectory_timeout_) {
      static double last_stale_warn_time = 0.0;
      double current_time_sec = current_time.seconds();
      if (current_time_sec - last_stale_warn_time >= 5.0) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "Trajectory is stale (%.2f s old, timeout=%.2f s), holding last position",
                    time_since_last_trajectory, trajectory_timeout_);
        last_stale_warn_time = current_time_sec;
      }
      target_q = latest_target_q_;
      trajectory_valid = true;
    } else {
      target_q = latest_target_q_;
      trajectory_valid = true;
    }
  }
  
  // Now we're outside the mutex, safe to access q_ and calculate distance
  if (!trajectory_valid) {
    return q_;
  }
  
  // Calculate distance to target
  Vector7d error = target_q - q_;
  double max_joint_error = error.cwiseAbs().maxCoeff();
  double euclidean_distance = error.norm();
  
  // Safety check: if distance exceeds threshold, use smooth approach
  if (max_joint_error > max_joint_distance_) {
    // Enter safe approach mode if not already in it, OR if a new far target was received
    // If a new far target arrived, reset the safe approach from current position
    bool was_reset = new_far_target_received_;
    if (!in_safe_approach_mode_ || was_reset) {
      in_safe_approach_mode_ = true;
      safe_approach_start_q_ = q_;  // Always start from CURRENT position for safety
      safe_approach_start_time_ = elapsed_time_;
      new_far_target_received_ = false;  // Clear the flag
      
      if (was_reset) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "New far target detected! Resetting safe approach from current position.");
      }
      RCLCPP_WARN(get_node()->get_logger(),
                  "Distance threshold exceeded! max_joint_error=%.4f rad (threshold=%.4f rad), "
                  "euclidean_distance=%.4f rad. Entering safe approach mode.",
                  max_joint_error, max_joint_distance_, euclidean_distance);
      RCLCPP_INFO(get_node()->get_logger(),
                  "Safe approach start position: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                  safe_approach_start_q_(0), safe_approach_start_q_(1), safe_approach_start_q_(2),
                  safe_approach_start_q_(3), safe_approach_start_q_(4), safe_approach_start_q_(5), safe_approach_start_q_(6));
      RCLCPP_INFO(get_node()->get_logger(),
                  "Target position: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                  target_q(0), target_q(1), target_q(2), target_q(3), target_q(4), target_q(5), target_q(6));
    }
    
    // Use minimum jerk to smoothly approach the target
    double safe_approach_elapsed = elapsed_time_ - safe_approach_start_time_;
    
    // Check if we've reached the target or completed the approach duration
    if (max_joint_error < 0.05 || safe_approach_elapsed >= safe_approach_duration_) {
      in_safe_approach_mode_ = false;
      if (max_joint_error < 0.05) {
        RCLCPP_INFO(get_node()->get_logger(),
                    "Reached target in safe approach mode. Final error: %.4f rad, time: %.2f s",
                    max_joint_error, safe_approach_elapsed);
      } else {
        RCLCPP_WARN(get_node()->get_logger(),
                    "Safe approach duration completed. Remaining error: %.4f rad, time: %.2f s",
                    max_joint_error, safe_approach_elapsed);
      }
      // Continue with normal trajectory following
      if (!reached_first_waypoint_) {
        reached_first_waypoint_ = true;
      }
      return target_q;
    }
    
    // Still in safe approach mode - use minimum jerk
    static double last_safe_debug_time = 0.0;
    if (elapsed_time_ - last_safe_debug_time >= 1.0) {
      RCLCPP_INFO(get_node()->get_logger(),
                  "Safe approach: t_elapsed=%.2f/%.2f s, max_error=%.4f rad, threshold=%.4f rad",
                  safe_approach_elapsed, safe_approach_duration_, max_joint_error, max_joint_distance_);
      last_safe_debug_time = elapsed_time_;
    }
    
    return computeMinimumJerkTrajectory(safe_approach_start_q_, target_q, 
                                        safe_approach_elapsed, safe_approach_duration_);
  }
  
  // Distance is within threshold - exit safe approach mode if we were in it
  if (in_safe_approach_mode_) {
    RCLCPP_INFO(get_node()->get_logger(),
                "Distance within threshold (max_error=%.4f < threshold=%.4f). Exiting safe approach mode.",
                max_joint_error, max_joint_distance_);
    in_safe_approach_mode_ = false;
  }
  
  // If we haven't reached the first waypoint, use minimum jerk to approach it
  if (!reached_first_waypoint_) {
    double t_elapsed = elapsed_time_ - approach_start_time_;
    
    // Debug logging every second
    static double last_debug_time = 0.0;
    if (elapsed_time_ - last_debug_time >= 1.0) {
      RCLCPP_INFO(get_node()->get_logger(), 
                  "Approaching first waypoint: t_elapsed=%.2f/%.2f, max_error=%.4f, threshold=0.05",
                  t_elapsed, approach_duration_, max_joint_error);
      RCLCPP_INFO(get_node()->get_logger(),
                  "Current q: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                  q_(0), q_(1), q_(2), q_(3), q_(4), q_(5), q_(6));
      RCLCPP_INFO(get_node()->get_logger(),
                  "Target q: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                  target_q(0), target_q(1), target_q(2), target_q(3), target_q(4), target_q(5), target_q(6));
      last_debug_time = elapsed_time_;
    }
    
    // Transition to trajectory following if time elapsed OR error is small enough
    if (t_elapsed >= approach_duration_ || max_joint_error < 0.05) {
      reached_first_waypoint_ = true;
      RCLCPP_INFO(get_node()->get_logger(), 
                  "Reached first waypoint! Starting trajectory following. "
                  "Final error: %.4f, Time elapsed: %.2f",
                  max_joint_error, t_elapsed);
    } else {
      // Still approaching first waypoint with minimum jerk
      return computeMinimumJerkTrajectory(initial_q_, target_q, t_elapsed, approach_duration_);
    }
  }
  
  // Once we've reached the first waypoint and distance is safe, directly follow the trajectory
  // This returns the LATEST received target position, which gets updated every time
  // a new trajectory message arrives in trajectoryCallback()
  return target_q;
}

controller_interface::return_type TrajectorySubscriberJointImpedanceController::update(
    const rclcpp::Time& time,
    const rclcpp::Duration& period) {
  static int update_count = 0;
  update_count++;
  
  updateJointStates();
  elapsed_time_ += period.seconds();
  
  // Debug logging every 100 updates (roughly every second at 1kHz)
  static double last_debug_time = 0.0;
  bool should_log = (elapsed_time_ - last_debug_time >= 1.0);
  
  if (should_log) {
    RCLCPP_INFO(get_node()->get_logger(), "=== UPDATE #%d (elapsed=%.2f s) ===", 
                update_count, elapsed_time_);
    RCLCPP_INFO(get_node()->get_logger(), 
                "Current joint positions: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                q_(0), q_(1), q_(2), q_(3), q_(4), q_(5), q_(6));
    RCLCPP_INFO(get_node()->get_logger(), 
                "Current joint velocities: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                dq_(0), dq_(1), dq_(2), dq_(3), dq_(4), dq_(5), dq_(6));
  }
  
  Vector7d q_goal = getCurrentTrajectoryWaypoint(time);
  
  if (should_log) {
    // Get latest target from trajectory for comparison
    Vector7d latest_target;
    bool has_target = false;
    {
      std::lock_guard<std::mutex> lock(trajectory_mutex_);
      if (trajectory_received_) {
        latest_target = latest_target_q_;
        has_target = true;
      }
    }
    
    RCLCPP_INFO(get_node()->get_logger(), 
                "Target joint positions (goal): [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                q_goal(0), q_goal(1), q_goal(2), q_goal(3), q_goal(4), q_goal(5), q_goal(6));
    if (has_target) {
      RCLCPP_INFO(get_node()->get_logger(),
                  "Latest received target: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                  latest_target(0), latest_target(1), latest_target(2),
                  latest_target(3), latest_target(4), latest_target(5), latest_target(6));
      Vector7d goal_diff = q_goal - latest_target;
      double max_diff = goal_diff.cwiseAbs().maxCoeff();
      if (max_diff > 0.001) {
        RCLCPP_DEBUG(get_node()->get_logger(),
                     "Goal differs from latest target by %.4f rad (due to smoothing/safety)",
                     max_diff);
      } else {
        RCLCPP_INFO(get_node()->get_logger(),
                    "Goal matches latest received target (controller tracking latest position)");
      }
    }
  }
  
  // Velocity filtering
  const double kAlpha = 0.99;
  dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;
  
  // Joint impedance control law
  Vector7d error = q_goal - q_;
  Vector7d tau_d_calculated =
      k_gains_.cwiseProduct(error) + d_gains_.cwiseProduct(-dq_filtered_);
  
  if (should_log) {
    double max_error = error.cwiseAbs().maxCoeff();
    double max_torque = tau_d_calculated.cwiseAbs().maxCoeff();
    RCLCPP_INFO(get_node()->get_logger(), 
                "Max position error: %.4f rad, reached_first_waypoint: %s",
                max_error, reached_first_waypoint_ ? "true" : "false");
    RCLCPP_INFO(get_node()->get_logger(), 
                "Computed torques: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                tau_d_calculated(0), tau_d_calculated(1), tau_d_calculated(2),
                tau_d_calculated(3), tau_d_calculated(4), tau_d_calculated(5), tau_d_calculated(6));
    RCLCPP_INFO(get_node()->get_logger(), "Max torque: %.3f Nm", max_torque);
    RCLCPP_INFO(get_node()->get_logger(), "Command interfaces count: %zu", command_interfaces_.size());
    last_debug_time = elapsed_time_;
  }
  
  // Set command values
  if (command_interfaces_.size() != static_cast<size_t>(num_joints)) {
    RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5.0,
                          "Command interfaces size mismatch: expected %d, got %zu",
                          num_joints, command_interfaces_.size());
    return controller_interface::return_type::ERROR;
  }
  
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
    if (should_log && i == 0) {
      RCLCPP_INFO(get_node()->get_logger(), 
                  "Set command_interface[%d] = %.3f (interface name: %s)",
                  i, tau_d_calculated(i), command_interfaces_[i].get_interface_name().c_str());
    }
  }
  
  return controller_interface::return_type::OK;
}

CallbackReturn TrajectorySubscriberJointImpedanceController::on_init() {
  try {
    auto_declare<bool>("gazebo", false);
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<std::string>("trajectory_topic", "arm_joint_trajectory");
    auto_declare<double>("approach_duration", 3.0);
    auto_declare<double>("trajectory_timeout", 1.0);
    auto_declare<double>("max_joint_distance", 0.5);  // Maximum joint distance (rad) before safe approach
    auto_declare<double>("safe_approach_duration", 2.0);  // Duration for safe approach when threshold exceeded
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn TrajectorySubscriberJointImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  is_gazebo = get_node()->get_parameter("gazebo").as_bool();
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  trajectory_topic_ = get_node()->get_parameter("trajectory_topic").as_string();
  approach_duration_ = get_node()->get_parameter("approach_duration").as_double();
  trajectory_timeout_ = get_node()->get_parameter("trajectory_timeout").as_double();
  max_joint_distance_ = get_node()->get_parameter("max_joint_distance").as_double();
  safe_approach_duration_ = get_node()->get_parameter("safe_approach_duration").as_double();
  
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
  
  // Get robot description
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
  
  // Create trajectory subscriber
  // Ensure topic name is absolute (starts with /) to avoid namespace issues
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
  // Use resolve_topic_name to ensure absolute topic works across namespaces
  std::string resolved_topic = absolute_topic;
  
  // For ROS 2, absolute topics (starting with /) should bypass namespace
  // But let's be explicit and ensure it's truly absolute
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
      std::bind(&TrajectorySubscriberJointImpedanceController::trajectoryCallback,
                this, std::placeholders::_1));
  
  RCLCPP_WARN(get_node()->get_logger(), 
              "Subscription created with topic: '%s' (should work across all namespaces)", 
              resolved_topic.c_str());
  
  // Log subscription details after creation
  RCLCPP_INFO(get_node()->get_logger(), "Subscription created successfully");
  RCLCPP_WARN(get_node()->get_logger(), 
              "=== VERIFICATION COMMANDS ===");
  RCLCPP_WARN(get_node()->get_logger(), 
              "1. Check topic exists: ros2 topic list | grep %s", absolute_topic.c_str());
  RCLCPP_WARN(get_node()->get_logger(), 
              "2. Check topic info: ros2 topic info %s", absolute_topic.c_str());
  RCLCPP_WARN(get_node()->get_logger(), 
              "3. Echo messages: ros2 topic echo %s", absolute_topic.c_str());
  RCLCPP_WARN(get_node()->get_logger(), 
              "4. Publisher MUST publish to exactly: '%s'", absolute_topic.c_str());
  RCLCPP_WARN(get_node()->get_logger(), 
              "5. Message type MUST be: trajectory_msgs/msg/JointTrajectory");
  
  RCLCPP_INFO(get_node()->get_logger(), 
              "=== TRAJECTORY SUBSCRIBER CONTROLLER CONFIGURED ===");
  RCLCPP_INFO(get_node()->get_logger(), "  Topic (configured): %s", trajectory_topic_.c_str());
  RCLCPP_INFO(get_node()->get_logger(), "  Topic (absolute): %s", absolute_topic.c_str());
  RCLCPP_INFO(get_node()->get_logger(), "  Node namespace: %s", node_namespace.c_str());
  RCLCPP_INFO(get_node()->get_logger(), "  Message type: trajectory_msgs/msg/JointTrajectory");
  RCLCPP_INFO(get_node()->get_logger(), "  Behavior: Controller will move to LATEST received joint position");
  RCLCPP_INFO(get_node()->get_logger(), "  arm_id: %s", arm_id_.c_str());
  RCLCPP_INFO(get_node()->get_logger(), "  is_gazebo: %s", is_gazebo ? "true" : "false");
  RCLCPP_INFO(get_node()->get_logger(), "  Approach duration: %.2f s", approach_duration_);
  RCLCPP_INFO(get_node()->get_logger(), "  Trajectory timeout: %.2f s", trajectory_timeout_);
  RCLCPP_INFO(get_node()->get_logger(), "  Max joint distance (safety): %.4f rad", max_joint_distance_);
  RCLCPP_INFO(get_node()->get_logger(), "  Safe approach duration: %.2f s", safe_approach_duration_);
  RCLCPP_INFO(get_node()->get_logger(), "  k_gains: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
              k_gains_(0), k_gains_(1), k_gains_(2), k_gains_(3), 
              k_gains_(4), k_gains_(5), k_gains_(6));
  RCLCPP_INFO(get_node()->get_logger(), "  d_gains: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
              d_gains_(0), d_gains_(1), d_gains_(2), d_gains_(3), 
              d_gains_(4), d_gains_(5), d_gains_(6));
  
  return CallbackReturn::SUCCESS;
}

CallbackReturn TrajectorySubscriberJointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  updateJointStates();
  dq_filtered_.setZero();
  initial_q_ = q_;
  elapsed_time_ = 0.0;
  approach_start_time_ = 0.0;
  reached_first_waypoint_ = false;
  in_safe_approach_mode_ = false;
  safe_approach_start_time_ = 0.0;
  new_far_target_received_ = false;
  new_far_target_received_ = false;
  
  // Reset trajectory state
  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    trajectory_received_ = false;
    latest_target_q_.setZero();
  }
  
  // Get absolute topic name
  std::string absolute_topic = trajectory_topic_;
  if (!absolute_topic.empty() && absolute_topic[0] != '/') {
    absolute_topic = "/" + absolute_topic;
  }
  
  RCLCPP_INFO(get_node()->get_logger(), 
              "=== TRAJECTORY SUBSCRIBER CONTROLLER ACTIVATED ===");
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
  RCLCPP_INFO(get_node()->get_logger(), "Initial position: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
              initial_q_(0), initial_q_(1), initial_q_(2), initial_q_(3),
              initial_q_(4), initial_q_(5), initial_q_(6));
  
  return CallbackReturn::SUCCESS;
}

void TrajectorySubscriberJointImpedanceController::updateJointStates() {
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

}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::TrajectorySubscriberJointImpedanceController,
                       controller_interface::ControllerInterface)

