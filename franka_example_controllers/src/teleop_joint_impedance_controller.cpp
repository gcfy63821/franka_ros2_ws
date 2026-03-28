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

#include <franka_example_controllers/teleop_joint_impedance_controller.hpp>
#include <franka_example_controllers/robot_utils.hpp>
#include <franka_example_controllers/default_robot_behavior_utils.hpp>

#include <cmath>
#include <string>

#include <Eigen/Eigen>

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
TeleopJointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints_; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
TeleopJointImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints_; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

CallbackReturn TeleopJointImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<std::string>("joint_command_topic", "/teleop_joint_commands");
    auto_declare<std::string>("gripper_command_topic", "/teleop_gripper_command");
    auto_declare<double>("command_timeout", 0.5);
    auto_declare<double>("max_joint_distance", 0.5);
    auto_declare<double>("gripper_threshold", 0.02);
    auto_declare<bool>("gazebo", false);

    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    if (arm_id_.empty()) {
      arm_id_ = "fr3";
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn TeleopJointImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  is_gazebo_ = get_node()->get_parameter("gazebo").as_bool();
  joint_command_topic_ = get_node()->get_parameter("joint_command_topic").as_string();
  gripper_command_topic_ = get_node()->get_parameter("gripper_command_topic").as_string();
  command_timeout_ = get_node()->get_parameter("command_timeout").as_double();
  max_joint_distance_ = get_node()->get_parameter("max_joint_distance").as_double();
  gripper_threshold_ = get_node()->get_parameter("gripper_threshold").as_double();

  // Read control gains
  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();

  if (k_gains.empty() || k_gains.size() != static_cast<size_t>(num_joints_)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "k_gains should be of size %d but is of size %zu", num_joints_, k_gains.size());
    return CallbackReturn::FAILURE;
  }
  if (d_gains.empty() || d_gains.size() != static_cast<size_t>(num_joints_)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "d_gains should be of size %d but is of size %zu", num_joints_, d_gains.size());
    return CallbackReturn::FAILURE;
  }

  for (int i = 0; i < num_joints_; ++i) {
    k_gains_(i) = k_gains.at(i);
    d_gains_(i) = d_gains.at(i);
  }
  dq_filtered_.setZero();

  // Get robot description for arm_id extraction
  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");

  if (parameters_client->wait_for_service(std::chrono::seconds(5))) {
    auto future = parameters_client->get_parameters({"robot_description"});
    auto result = future.get();
    if (!result.empty()) {
      robot_description_ = result[0].value_to_string();
      auto extracted_id =
          robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());
      if (!extracted_id.empty()) {
        arm_id_ = extracted_id;
      }
    }
  }

  if (arm_id_.empty()) {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
  }

  RCLCPP_INFO(get_node()->get_logger(), "arm_id: '%s'", arm_id_.c_str());

  // Get namespace for gripper actions
  namespace_ = get_node()->get_namespace();

  // Create gripper action clients (only for real hardware)
  if (!is_gazebo_) {
    std::string ns(namespace_);
    if (ns == "/") ns = "";
    std::string grasp_action = ns + "/franka_gripper/grasp";
    std::string move_action = ns + "/franka_gripper/move";

    gripper_grasp_client_ = rclcpp_action::create_client<franka_msgs::action::Grasp>(
        get_node(), grasp_action);
    gripper_move_client_ = rclcpp_action::create_client<franka_msgs::action::Move>(
        get_node(), move_action);

    RCLCPP_INFO(get_node()->get_logger(), "Gripper action clients: grasp='%s', move='%s'",
                grasp_action.c_str(), move_action.c_str());
  } else {
    RCLCPP_INFO(get_node()->get_logger(),
                "Gazebo mode: Gripper action clients not created (gripper control disabled)");
  }

  // Set default collision behavior (only for real hardware)
  if (!is_gazebo_) {
    auto client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
        "service_server/set_full_collision_behavior");
    auto request = DefaultRobotBehavior::getDefaultCollisionBehaviorRequest();
    auto future_result = client->async_send_request(request);
    future_result.wait_for(robot_utils::time_out);
    auto success = future_result.get();
    if (!success) {
      RCLCPP_FATAL(get_node()->get_logger(), "Failed to set default collision behavior.");
      return CallbackReturn::ERROR;
    }
    RCLCPP_INFO(get_node()->get_logger(), "Default collision behavior set.");
  } else {
    RCLCPP_INFO(get_node()->get_logger(),
                "Gazebo mode: Skipping collision behavior setup");
  }

  // Create joint command subscription (BestEffort QoS for low latency)
  rclcpp::QoS qos_profile(1);
  qos_profile.reliability(rclcpp::ReliabilityPolicy::BestEffort);
  qos_profile.durability(rclcpp::DurabilityPolicy::Volatile);

  joint_command_sub_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      joint_command_topic_, qos_profile,
      std::bind(&TeleopJointImpedanceController::jointCommandCallback,
                this, std::placeholders::_1));

  // Create gripper command subscription
  gripper_command_sub_ = get_node()->create_subscription<std_msgs::msg::Float64>(
      gripper_command_topic_, qos_profile,
      std::bind(&TeleopJointImpedanceController::gripperCommandCallback,
                this, std::placeholders::_1));

  RCLCPP_INFO(get_node()->get_logger(),
              "Subscriptions created: joints='%s', gripper='%s'",
              joint_command_topic_.c_str(), gripper_command_topic_.c_str());
  RCLCPP_INFO(get_node()->get_logger(),
              "Gains K=[%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f], D=[%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f]",
              k_gains_(0), k_gains_(1), k_gains_(2), k_gains_(3),
              k_gains_(4), k_gains_(5), k_gains_(6),
              d_gains_(0), d_gains_(1), d_gains_(2), d_gains_(3),
              d_gains_(4), d_gains_(5), d_gains_(6));

  return CallbackReturn::SUCCESS;
}

CallbackReturn TeleopJointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  // Verify interfaces
  if (command_interfaces_.size() != static_cast<size_t>(num_joints_)) {
    RCLCPP_FATAL(get_node()->get_logger(), "Expected %d command interfaces, got %zu",
                 num_joints_, command_interfaces_.size());
    return CallbackReturn::ERROR;
  }
  if (state_interfaces_.size() != static_cast<size_t>(2 * num_joints_)) {
    RCLCPP_FATAL(get_node()->get_logger(), "Expected %d state interfaces, got %zu",
                 2 * num_joints_, state_interfaces_.size());
    return CallbackReturn::ERROR;
  }

  // Read current position and hold it
  updateJointStates();
  q_target_ = q_;
  initial_q_ = q_;
  dq_filtered_.setZero();
  elapsed_time_ = 0.0;
  last_command_elapsed_ = 0.0;
  command_received_ = false;
  timeout_warned_ = false;
  gripper_is_open_ = true;
  gripper_action_pending_ = false;

  // Wait for gripper action servers (only for real hardware)
  if (!is_gazebo_) {
    if (gripper_move_client_ &&
        !gripper_move_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Gripper Move action server not available (gripper control may not work)");
    }
    if (gripper_grasp_client_ &&
        !gripper_grasp_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Gripper Grasp action server not available (gripper control may not work)");
    }
  }

  RCLCPP_INFO(get_node()->get_logger(),
              "=== TELEOP CONTROLLER ACTIVATED (gazebo=%s) ===",
              is_gazebo_ ? "true" : "false");
  RCLCPP_INFO(get_node()->get_logger(),
              "Holding current position: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
              initial_q_(0), initial_q_(1), initial_q_(2), initial_q_(3),
              initial_q_(4), initial_q_(5), initial_q_(6));
  RCLCPP_INFO(get_node()->get_logger(),
              "Waiting for commands on '%s' (sensor_msgs/JointState)",
              joint_command_topic_.c_str());
  RCLCPP_INFO(get_node()->get_logger(),
              "Gripper commands on '%s' (std_msgs/Float64, width in meters)",
              gripper_command_topic_.c_str());
  RCLCPP_INFO(get_node()->get_logger(),
              "Command timeout: %.2f s, max joint distance: %.3f rad",
              command_timeout_, max_joint_distance_);

  return CallbackReturn::SUCCESS;
}

controller_interface::return_type TeleopJointImpedanceController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  updateJointStates();
  elapsed_time_ += period.seconds();

  // Velocity filtering (alpha = 0.99, light filter)
  const double kAlpha = 0.99;
  dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;

  // Get current target (thread-safe)
  Vector7d target;
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    target = q_target_;
  }

  // Command timeout warning (throttled)
  if (command_received_ && (elapsed_time_ - last_command_elapsed_) > command_timeout_) {
    if (!timeout_warned_) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "No teleop commands received for %.1f s, holding position",
                  command_timeout_);
      timeout_warned_ = true;
    }
  }

  // Joint impedance control: tau = k * (q_target - q) + d * (-dq_filtered)
  Vector7d tau_d = k_gains_.cwiseProduct(target - q_) + d_gains_.cwiseProduct(-dq_filtered_);

  for (int i = 0; i < num_joints_; ++i) {
    command_interfaces_[i].set_value(tau_d(i));
  }

  // Periodic debug output (~1Hz)
  if (command_received_ && static_cast<int>(elapsed_time_) > static_cast<int>(elapsed_time_ - period.seconds())) {
    Vector7d err = target - q_;
    RCLCPP_INFO(get_node()->get_logger(),
                "[TELEOP] q=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f] "
                "err=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] "
                "tau=[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]",
                q_(0), q_(1), q_(2), q_(3), q_(4), q_(5), q_(6),
                err(0), err(1), err(2), err(3), err(4), err(5), err(6),
                tau_d(0), tau_d(1), tau_d(2), tau_d(3), tau_d(4), tau_d(5), tau_d(6));
  }

  return controller_interface::return_type::OK;
}

void TeleopJointImpedanceController::updateJointStates() {
  for (int i = 0; i < num_joints_; ++i) {
    q_(i) = state_interfaces_.at(2 * i).get_value();
    dq_(i) = state_interfaces_.at(2 * i + 1).get_value();
  }
}

void TeleopJointImpedanceController::jointCommandCallback(
    const sensor_msgs::msg::JointState::SharedPtr msg) {
  if (msg->position.size() < static_cast<size_t>(num_joints_)) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                         "Joint command has %zu positions, expected %d",
                         msg->position.size(), num_joints_);
    return;
  }

  Vector7d new_target;
  for (int i = 0; i < num_joints_; ++i) {
    new_target(i) = msg->position[i];
  }

  // Safety: reject if any joint target is too far from current target
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    if (command_received_) {
      double max_dist = (new_target - q_target_).cwiseAbs().maxCoeff();
      if (max_dist > max_joint_distance_) {
        RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                             "Rejecting command: max joint jump %.4f rad > limit %.4f rad",
                             max_dist, max_joint_distance_);
        return;
      }
    }
    q_target_ = new_target;
    command_received_ = true;
    last_command_elapsed_ = elapsed_time_;
    timeout_warned_ = false;
  }
}

void TeleopJointImpedanceController::gripperCommandCallback(
    const std_msgs::msg::Float64::SharedPtr msg) {
  if (is_gazebo_) {
    return;  // Gripper not available in Gazebo
  }

  double width = msg->data;
  bool should_close = (width < gripper_threshold_);

  if (should_close && gripper_is_open_ && !gripper_action_pending_) {
    closeGripper();
  } else if (!should_close && !gripper_is_open_ && !gripper_action_pending_) {
    openGripper();
  }
}

void TeleopJointImpedanceController::openGripper() {
  if (!gripper_move_client_ || !gripper_move_client_->action_server_is_ready()) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5000,
                         "Gripper Move action server not ready");
    return;
  }

  franka_msgs::action::Move::Goal goal;
  goal.width = 0.08;
  goal.speed = 0.1;

  auto send_goal_options = rclcpp_action::Client<franka_msgs::action::Move>::SendGoalOptions();
  send_goal_options.result_callback =
      [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Move>::WrappedResult&) {
        gripper_action_pending_ = false;
        gripper_is_open_ = true;
        RCLCPP_INFO(get_node()->get_logger(), "Gripper opened");
      };

  gripper_action_pending_ = true;
  gripper_move_client_->async_send_goal(goal, send_goal_options);
  RCLCPP_INFO(get_node()->get_logger(), "Opening gripper...");
}

void TeleopJointImpedanceController::closeGripper() {
  if (!gripper_grasp_client_ || !gripper_grasp_client_->action_server_is_ready()) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 5000,
                         "Gripper Grasp action server not ready");
    return;
  }

  franka_msgs::action::Grasp::Goal goal;
  goal.width = 0.0;
  goal.speed = 0.1;
  goal.force = 40.0;
  goal.epsilon.inner = 0.04;
  goal.epsilon.outer = 0.04;

  auto send_goal_options = rclcpp_action::Client<franka_msgs::action::Grasp>::SendGoalOptions();
  send_goal_options.result_callback =
      [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Grasp>::WrappedResult&) {
        gripper_action_pending_ = false;
        gripper_is_open_ = false;
        RCLCPP_INFO(get_node()->get_logger(), "Gripper closed");
      };

  gripper_action_pending_ = true;
  gripper_grasp_client_->async_send_goal(goal, send_goal_options);
  RCLCPP_INFO(get_node()->get_logger(), "Closing gripper (force: 40N)...");
}

}  // namespace franka_example_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::TeleopJointImpedanceController,
                       controller_interface::ControllerInterface)
