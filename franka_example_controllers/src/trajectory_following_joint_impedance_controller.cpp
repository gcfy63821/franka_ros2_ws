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

#include <franka_example_controllers/trajectory_following_joint_impedance_controller.hpp>
#include <franka_example_controllers/robot_utils.hpp>
#include <franka_example_controllers/default_robot_behavior_utils.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

#include <Eigen/Eigen>

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
TrajectoryFollowingJointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
TrajectoryFollowingJointImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

TrajectoryFollowingJointImpedanceController::Vector7d 
TrajectoryFollowingJointImpedanceController::computeMinimumJerkTrajectory(
    const TrajectoryFollowingJointImpedanceController::Vector7d& q_start, 
    const TrajectoryFollowingJointImpedanceController::Vector7d& q_end, 
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

TrajectoryFollowingJointImpedanceController::Vector7d 
TrajectoryFollowingJointImpedanceController::computeSmoothTrajectoryWaypoint(
    size_t frame_index, double frame_fraction) {
  if (frame_index >= trajectory_.size()) {
    return trajectory_.back();
  }
  
  // If we're at the last frame, just return it
  if (frame_index >= trajectory_.size() - 1) {
    return trajectory_.back();
  }
  
  // Get current and next frame
  const Vector7d& q_current = trajectory_[frame_index];
  const Vector7d& q_next = trajectory_[frame_index + 1];
  
  // Use cubic interpolation for smooth velocity
  // This creates a smooth transition with continuous velocity
  double s = std::max(0.0, std::min(1.0, frame_fraction));
  
  // Cubic Hermite interpolation: smooth with continuous velocity
  // h1(t) = 2t^3 - 3t^2 + 1
  // h2(t) = -2t^3 + 3t^2
  double s2 = s * s;
  double s3 = s2 * s;
  double h1 = 2.0 * s3 - 3.0 * s2 + 1.0;
  double h2 = -2.0 * s3 + 3.0 * s2;
  
  // If we have future frames, use them to estimate velocity for smoother motion
  Vector7d velocity_current = Vector7d::Zero();
  Vector7d velocity_next = Vector7d::Zero();
  double dt = 1.0 / trajectory_rate_;
  
  if (frame_index > 0 && frame_index < trajectory_.size() - 1) {
    // Estimate velocity at current frame using previous and next frames
    // Central difference: v = (q_next - q_prev) / (2*dt)
    const Vector7d& q_prev = trajectory_[frame_index - 1];
    velocity_current = (q_next - q_prev) / (2.0 * dt);
  } else if (frame_index == 0) {
    // At first frame, use forward difference
    velocity_current = (q_next - q_current) / dt;
  }
  
  if (frame_index < trajectory_.size() - 2) {
    // Estimate velocity at next frame using current and future frames
    // Central difference: v = (q_future - q_current) / (2*dt)
    const Vector7d& q_future = trajectory_[frame_index + 2];
    velocity_next = (q_future - q_current) / (2.0 * dt);
  } else if (frame_index == trajectory_.size() - 2) {
    // At second-to-last frame, use backward difference
    velocity_next = (q_next - q_current) / dt;
  }
  
  // Cubic Hermite interpolation with velocity
  // p(t) = h1 * p0 + h2 * p1 + h3 * v0*T + h4 * v1*T
  // where h3 = t^3 - 2t^2 + t, h4 = t^3 - t^2, T = dt
  double h3 = s3 - 2.0 * s2 + s;
  double h4 = s3 - s2;
  
  Vector7d interpolated = h1 * q_current + h2 * q_next + 
                          h3 * velocity_current * dt + 
                          h4 * velocity_next * dt;
  
  return interpolated;
}

void TrajectoryFollowingJointImpedanceController::loadTrajectoryFromYAML(
    const std::string& file_path) {
  trajectory_.clear();
  
  std::ifstream file(file_path);
  if (!file.is_open()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to open trajectory file: %s", file_path.c_str());
    return;
  }
  
  std::string line;
  while (std::getline(file, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }
    
    std::istringstream iss(line);
    Vector7d waypoint;
    bool valid = true;
    
    for (int i = 0; i < num_joints; ++i) {
      if (!(iss >> waypoint(i))) {
        valid = false;
        break;
      }
    }
    
    if (valid) {
      trajectory_.push_back(waypoint);
    } else {
      RCLCPP_WARN(get_node()->get_logger(), "Invalid waypoint in trajectory file, skipping line");
    }
  }
  
  file.close();
  
  if (trajectory_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No valid waypoints loaded from trajectory file");
    return;
  }
  
  trajectory_loaded_ = true;
  RCLCPP_INFO(get_node()->get_logger(), "Loaded %zu waypoints from trajectory file",
              trajectory_.size());
}

TrajectoryFollowingJointImpedanceController::Vector7d 
TrajectoryFollowingJointImpedanceController::getCurrentTrajectoryWaypoint() {
  if (!trajectory_loaded_ || trajectory_.empty()) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5.0,
                         "Trajectory not loaded or empty");
    return q_;
  }
  
  // If we haven't reached the first waypoint, use minimum jerk to approach it
  if (!reached_first_waypoint_) {
    double t_elapsed = elapsed_time_ - approach_start_time_;
    
    // Check if we've reached the first waypoint
    Vector7d error = trajectory_[0] - q_;
    double max_error = error.cwiseAbs().maxCoeff();
    
    // Debug logging every second
    static double last_debug_time = 0.0;
    if (elapsed_time_ - last_debug_time >= 1.0) {
      RCLCPP_INFO(get_node()->get_logger(), 
                  "Approaching first waypoint: t_elapsed=%.2f/%.2f, max_error=%.4f, threshold=0.05",
                  t_elapsed, approach_duration_, max_error);
      RCLCPP_INFO(get_node()->get_logger(),
                  "Current q: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                  q_(0), q_(1), q_(2), q_(3), q_(4), q_(5), q_(6));
      RCLCPP_INFO(get_node()->get_logger(),
                  "Target q: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                  trajectory_[0](0), trajectory_[0](1), trajectory_[0](2),
                  trajectory_[0](3), trajectory_[0](4), trajectory_[0](5), trajectory_[0](6));
      last_debug_time = elapsed_time_;
    }
    
    // Transition to trajectory following if time elapsed OR error is small enough
    if (t_elapsed >= approach_duration_ || max_error < 0.05) {
      reached_first_waypoint_ = true;
      current_trajectory_index_ = 0;
      // Reset elapsed_time_ to start trajectory following from time 0
      elapsed_time_ = 0.0;
      RCLCPP_INFO(get_node()->get_logger(), 
                  "Reached first waypoint! Starting trajectory following. "
                  "Final error: %.4f, Time elapsed: %.2f",
                  max_error, t_elapsed);
      RCLCPP_INFO(get_node()->get_logger(), 
                  "Trajectory has %zu waypoints, following at %.1f Hz (%.3f s per frame)",
                  trajectory_.size(), trajectory_rate_, 1.0 / trajectory_rate_);
      RCLCPP_INFO(get_node()->get_logger(),
                  "Expected trajectory duration: %.2f seconds",
                  trajectory_.size() / trajectory_rate_);
    } else {
      // Still approaching first waypoint with minimum jerk
      Vector7d first_waypoint = trajectory_[0];
      return computeMinimumJerkTrajectory(initial_q_, first_waypoint, t_elapsed, approach_duration_);
    }
  }
  
  // Follow trajectory at specified rate with smooth interpolation
  if (reached_first_waypoint_) {
    double trajectory_time = elapsed_time_;
    // Calculate frame index and fractional part for smooth interpolation
    // At 30 Hz, each frame is 1/30 = 0.0333 seconds
    double frame_time = trajectory_time * trajectory_rate_;
    size_t frame_index = static_cast<size_t>(std::floor(frame_time));
    double frame_fraction = frame_time - frame_index;  // Fraction within current frame [0, 1)
    
    // Debug logging to verify 30 Hz frame progression
    static double last_traj_debug_time = -1.0;
    static size_t last_frame_index = SIZE_MAX;
    static double last_frame_change_time = -1.0;
    
    // Log every frame change to verify 30 Hz
    bool frame_changed = (last_frame_index != SIZE_MAX && frame_index != last_frame_index);
    bool time_to_log = (last_traj_debug_time < 0 || elapsed_time_ - last_traj_debug_time >= 0.1);
    
    if (frame_changed) {
      double time_since_last_frame = elapsed_time_ - last_frame_change_time;
      double actual_frame_rate = 1.0 / time_since_last_frame;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Frame advanced: %zu -> %zu, time=%.3f, dt=%.4f, rate=%.1f Hz (target=%.1f Hz)",
                  last_frame_index, frame_index, trajectory_time, 
                  time_since_last_frame, actual_frame_rate, trajectory_rate_);
      last_frame_change_time = elapsed_time_;
    }
    
    if (time_to_log) {
      double expected_frame_time = frame_index / trajectory_rate_;
      RCLCPP_DEBUG(get_node()->get_logger(),
                  "Trajectory following: time=%.3f, frame=%zu/%zu (frac=%.3f), expected_time=%.3f",
                  trajectory_time, frame_index, trajectory_.size(), frame_fraction, expected_frame_time);
      last_traj_debug_time = elapsed_time_;
    }
    
    if (last_frame_index == SIZE_MAX || frame_index != last_frame_index) {
      last_frame_index = frame_index;
      if (last_frame_change_time < 0) {
        last_frame_change_time = elapsed_time_;
      }
    }
    
    if (frame_index >= trajectory_.size()) {
      // Reached end of trajectory, hold last waypoint
      if (frame_index == trajectory_.size()) {
        RCLCPP_INFO(get_node()->get_logger(), 
                    "Reached end of trajectory at time %.2f, holding last waypoint",
                    trajectory_time);
      }
      return trajectory_.back();
    }
    
    current_trajectory_index_ = frame_index;
    
    // Use smooth interpolation between frames
    return computeSmoothTrajectoryWaypoint(frame_index, frame_fraction);
  }
  
  return q_;
}

controller_interface::return_type TrajectoryFollowingJointImpedanceController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  updateJointStates();
  elapsed_time_ += period.seconds();
  
  Vector7d q_goal = getCurrentTrajectoryWaypoint();
  
  // Velocity filtering
  const double kAlpha = 0.99;
  dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;
  
  // Joint impedance control law
  Vector7d error = q_goal - q_;
  Vector7d tau_d_calculated =
      k_gains_.cwiseProduct(error) + d_gains_.cwiseProduct(-dq_filtered_);
  
  // Debug logging for control errors (throttled)
  static double last_control_debug_time = 0.0;
  if (elapsed_time_ - last_control_debug_time >= 2.0) {
    double max_error = error.cwiseAbs().maxCoeff();
    RCLCPP_DEBUG(get_node()->get_logger(),
                 "Control update: max_error=%.4f, reached_first=%d, elapsed=%.2f",
                 max_error, reached_first_waypoint_, elapsed_time_);
    last_control_debug_time = elapsed_time_;
  }
  
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }
  
  return controller_interface::return_type::OK;
}

CallbackReturn TrajectoryFollowingJointImpedanceController::on_init() {
  try {
    auto_declare<bool>("gazebo", false);
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::string>("trajectory_file", "");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<double>("trajectory_rate", 30.0);
    auto_declare<double>("approach_duration", 3.0);
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn TrajectoryFollowingJointImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  is_gazebo = get_node()->get_parameter("gazebo").as_bool();
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  trajectory_file_path_ = get_node()->get_parameter("trajectory_file").as_string();
  trajectory_rate_ = get_node()->get_parameter("trajectory_rate").as_double();
  approach_duration_ = get_node()->get_parameter("approach_duration").as_double();
  trajectory_dt_ = 1.0 / trajectory_rate_;
  
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
  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();
  
  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }
  
  arm_id_ = robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());
  
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
  
  // Load trajectory
  if (trajectory_file_path_.empty()) {
    RCLCPP_WARN(get_node()->get_logger(), "No trajectory file specified, controller will hold current position");
  } else {
    RCLCPP_INFO(get_node()->get_logger(), "Loading trajectory from: %s", trajectory_file_path_.c_str());
    loadTrajectoryFromYAML(trajectory_file_path_);
    if (trajectory_loaded_) {
      RCLCPP_INFO(get_node()->get_logger(), 
                  "Trajectory loaded successfully: %zu waypoints, rate=%.1f Hz, approach_duration=%.1f s",
                  trajectory_.size(), trajectory_rate_, approach_duration_);
    } else {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to load trajectory from: %s", trajectory_file_path_.c_str());
    }
  }
  
  return CallbackReturn::SUCCESS;
}

CallbackReturn TrajectoryFollowingJointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  updateJointStates();
  dq_filtered_.setZero();
  initial_q_ = q_;
  elapsed_time_ = 0.0;
  approach_start_time_ = 0.0;
  reached_first_waypoint_ = false;
  current_trajectory_index_ = 0;
  
  if (trajectory_loaded_ && !trajectory_.empty()) {
    double expected_duration = trajectory_.size() / trajectory_rate_;
    RCLCPP_INFO(get_node()->get_logger(), 
                "Starting trajectory following. Trajectory rate: %.1f Hz (%.4f s per frame)",
                trajectory_rate_, 1.0 / trajectory_rate_);
    RCLCPP_INFO(get_node()->get_logger(), 
                "Expected trajectory duration: %.2f seconds (%zu frames)",
                expected_duration, trajectory_.size());
    RCLCPP_INFO(get_node()->get_logger(), "Initial position: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                initial_q_(0), initial_q_(1), initial_q_(2), initial_q_(3),
                initial_q_(4), initial_q_(5), initial_q_(6));
    RCLCPP_INFO(get_node()->get_logger(), "First waypoint: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                trajectory_[0](0), trajectory_[0](1), trajectory_[0](2), trajectory_[0](3),
                trajectory_[0](4), trajectory_[0](5), trajectory_[0](6));
  }
  
  return CallbackReturn::SUCCESS;
}

void TrajectoryFollowingJointImpedanceController::updateJointStates() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);
    
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");
    
    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::TrajectoryFollowingJointImpedanceController,
                       controller_interface::ControllerInterface)

