// Copyright (c) 2026 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <franka_example_controllers/ee_pose_replay_controller.hpp>

#include <franka_example_controllers/default_robot_behavior_utils.hpp>
#include <franka_example_controllers/robot_utils.hpp>

#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>

using namespace std::chrono_literals;

namespace {

double minJerkPositionScale(double t, double duration) {
  if (duration <= 1e-9) {
    return 1.0;
  }
  const double u = std::clamp(t / duration, 0.0, 1.0);
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  const double u5 = u4 * u;
  return 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
}

Eigen::Quaterniond normalizedQuaternion(double w, double x, double y, double z) {
  Eigen::Quaterniond q(w, x, y, z);
  if (q.norm() <= 1e-9) {
    return Eigen::Quaterniond::Identity();
  }
  q.normalize();
  return q;
}

// Damped right pseudo-inverse for a wide Jacobian (6x7) used in nullspace
// projection. damping = 0.05 is conservative for Franka working space.
Eigen::Matrix<double, 7, 6> dampedPseudoInverse(const Eigen::Matrix<double, 6, 7>& J,
                                                double damping = 0.05) {
  Eigen::Matrix<double, 6, 6> JJt = J * J.transpose();
  JJt.diagonal().array() += damping * damping;
  return J.transpose() * JJt.inverse();
}

}  // namespace

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
EePoseReplayController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
EePoseReplayController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  for (const auto& name : franka_cartesian_pose_->get_state_interface_names()) {
    config.names.push_back(name);
  }
  return config;
}

CallbackReturn EePoseReplayController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::string>("base_link", "fr3_link0");
    auto_declare<std::string>("ee_link", "fr3_hand_tcp");
    auto_declare<bool>("gazebo", false);
    auto_declare<std::string>("mode_topic", "/ee_pose_replay/mode");
    auto_declare<std::string>("trajectory_topic", "/ee_pose_replay/trajectory");
    auto_declare<std::string>("start_topic", "/ee_pose_replay/replay_started");
    auto_declare<std::string>("finish_topic", "/ee_pose_replay/replay_finished");
    auto_declare<bool>("move_to_start", true);
    auto_declare<double>("move_to_start_min_duration", 4.0);
    auto_declare<double>("move_to_start_max_translation_velocity", 0.05);
    auto_declare<double>("move_to_start_max_rotation_velocity", 0.5);
    auto_declare<double>("translational_stiffness", 200.0);
    auto_declare<double>("rotational_stiffness", 20.0);
    auto_declare<double>("nullspace_stiffness", 10.0);
    auto_declare<double>("joint_damping_ratio", 1.0);
  } catch (const std::exception& e) {
    fprintf(stderr, "EePoseReplayController init failed: %s\n", e.what());
    return CallbackReturn::ERROR;
  }

  franka_cartesian_pose_ =
      std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(
          franka_semantic_components::FrankaCartesianPoseInterface(k_elbow_activated_));
  return CallbackReturn::SUCCESS;
}

bool EePoseReplayController::setupKdlFromUrdf() {
  // Fetch robot_description from /robot_state_publisher (same pattern as
  // joint_impedance_with_ik_example_controller).
  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  if (!parameters_client->wait_for_service(5s)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "robot_state_publisher parameter service unavailable; can't read URDF");
    return false;
  }
  auto future = parameters_client->get_parameters({"robot_description"});
  if (future.wait_for(5s) != std::future_status::ready) {
    RCLCPP_FATAL(get_node()->get_logger(), "Timed out waiting for robot_description");
    return false;
  }
  auto result = future.get();
  if (result.empty() || result[0].value_to_string().empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "robot_description is empty");
    return false;
  }
  robot_description_ = result[0].value_to_string();

  KDL::Tree tree;
  if (!kdl_parser::treeFromString(robot_description_, tree)) {
    RCLCPP_FATAL(get_node()->get_logger(), "Failed to parse URDF into KDL tree");
    return false;
  }
  if (!tree.getChain(base_link_, ee_link_, kdl_chain_)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "Failed to extract KDL chain from '%s' to '%s'. Available links may differ; "
                 "check the URDF.",
                 base_link_.c_str(), ee_link_.c_str());
    return false;
  }
  if (kdl_chain_.getNrOfJoints() != static_cast<unsigned int>(kNumJoints)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "KDL chain has %u movable joints, expected %d. Check base_link/ee_link.",
                 kdl_chain_.getNrOfJoints(), kNumJoints);
    return false;
  }
  jac_solver_ = std::make_unique<KDL::ChainJntToJacSolver>(kdl_chain_);
  kdl_q_.resize(kdl_chain_.getNrOfJoints());
  kdl_jac_.resize(kdl_chain_.getNrOfJoints());

  RCLCPP_INFO(get_node()->get_logger(),
              "KDL chain built: base='%s' ee='%s' nr_joints=%u nr_segments=%u",
              base_link_.c_str(), ee_link_.c_str(), kdl_chain_.getNrOfJoints(),
              kdl_chain_.getNrOfSegments());
  return true;
}

EePoseReplayController::Matrix6x7d EePoseReplayController::computeJacobian() const {
  for (int i = 0; i < kNumJoints; ++i) {
    kdl_q_(i) = q_(i);
  }
  jac_solver_->JntToJac(kdl_q_, kdl_jac_);
  // KDL Jacobian rows: 0-2 linear, 3-5 angular, in the chain's root frame (= base_link_).
  Matrix6x7d J;
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < kNumJoints; ++c) {
      J(r, c) = kdl_jac_(r, c);
    }
  }
  return J;
}

CallbackReturn EePoseReplayController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  base_link_ = get_node()->get_parameter("base_link").as_string();
  ee_link_ = get_node()->get_parameter("ee_link").as_string();
  gazebo_ = get_node()->get_parameter("gazebo").as_bool();
  mode_topic_ = get_node()->get_parameter("mode_topic").as_string();
  trajectory_topic_ = get_node()->get_parameter("trajectory_topic").as_string();
  start_topic_ = get_node()->get_parameter("start_topic").as_string();
  finish_topic_ = get_node()->get_parameter("finish_topic").as_string();
  move_to_start_enabled_ = get_node()->get_parameter("move_to_start").as_bool();
  move_to_start_min_duration_ =
      get_node()->get_parameter("move_to_start_min_duration").as_double();
  move_to_start_max_translation_velocity_ =
      get_node()->get_parameter("move_to_start_max_translation_velocity").as_double();
  move_to_start_max_rotation_velocity_ =
      get_node()->get_parameter("move_to_start_max_rotation_velocity").as_double();
  translational_stiffness_ = get_node()->get_parameter("translational_stiffness").as_double();
  rotational_stiffness_ = get_node()->get_parameter("rotational_stiffness").as_double();
  nullspace_stiffness_ = get_node()->get_parameter("nullspace_stiffness").as_double();
  joint_damping_ratio_ = get_node()->get_parameter("joint_damping_ratio").as_double();

  if (move_to_start_max_translation_velocity_ <= 0.0 ||
      move_to_start_max_rotation_velocity_ <= 0.0) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "move_to_start max translation/rotation velocities must be > 0");
    return CallbackReturn::FAILURE;
  }

  cartesian_stiffness_.setZero();
  cartesian_stiffness_.topLeftCorner(3, 3) =
      translational_stiffness_ * Eigen::Matrix3d::Identity();
  cartesian_stiffness_.bottomRightCorner(3, 3) =
      rotational_stiffness_ * Eigen::Matrix3d::Identity();
  cartesian_damping_.setZero();
  cartesian_damping_.topLeftCorner(3, 3) =
      2.0 * joint_damping_ratio_ * std::sqrt(translational_stiffness_) *
      Eigen::Matrix3d::Identity();
  cartesian_damping_.bottomRightCorner(3, 3) =
      2.0 * joint_damping_ratio_ * std::sqrt(rotational_stiffness_) *
      Eigen::Matrix3d::Identity();

  if (!setupKdlFromUrdf()) {
    return CallbackReturn::ERROR;
  }

  if (!gazebo_) {
    auto client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
        "service_server/set_full_collision_behavior");
    if (!client->wait_for_service(robot_utils::time_out)) {
      RCLCPP_FATAL(get_node()->get_logger(), "Collision behavior service unavailable.");
      return CallbackReturn::ERROR;
    }
    auto future_result =
        client->async_send_request(DefaultRobotBehavior::getDefaultCollisionBehaviorRequest());
    if (future_result.wait_for(robot_utils::time_out) != std::future_status::ready ||
        !future_result.get()) {
      RCLCPP_FATAL(get_node()->get_logger(), "Failed to set default collision behavior.");
      return CallbackReturn::ERROR;
    }
    RCLCPP_INFO(get_node()->get_logger(), "Default collision behavior set.");
  }

  mode_sub_ = get_node()->create_subscription<std_msgs::msg::String>(
      mode_topic_, rclcpp::QoS(10),
      std::bind(&EePoseReplayController::modeCallback, this, std::placeholders::_1));
  trajectory_sub_ =
      get_node()->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>(
          trajectory_topic_, rclcpp::QoS(1),
          std::bind(&EePoseReplayController::trajectoryCallback, this, std::placeholders::_1));
  start_pub_ = get_node()->create_publisher<std_msgs::msg::Bool>(start_topic_, rclcpp::QoS(10));
  finish_pub_ = get_node()->create_publisher<std_msgs::msg::Bool>(finish_topic_, rclcpp::QoS(10));

  RCLCPP_INFO(get_node()->get_logger(),
              "EePoseReplayController (Cartesian impedance, KDL Jacobian) configured. "
              "K_t=%.1f K_r=%.1f K_n=%.1f damping_ratio=%.2f",
              translational_stiffness_, rotational_stiffness_, nullspace_stiffness_,
              joint_damping_ratio_);
  return CallbackReturn::SUCCESS;
}

CallbackReturn EePoseReplayController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_);

  // Read-once happens after on_activate, so capture initial state on the first
  // update() tick (via needs_initialization_ flag) instead of here.
  needs_initialization_ = true;
  dq_filtered_.setZero();
  hold_initialized_ = false;
  last_command_initialized_ = false;

  phase_.store(static_cast<int>(Phase::HOLD));
  active_trajectory_.reset();
  next_trajectory_.reset();
  has_new_trajectory_.store(false);
  start_pending_ = false;
  finish_pending_ = false;
  tracking_initialized_ = false;
  hold_reset_requested_.store(false);
  replay_elapsed_ = 0.0;
  segment_index_ = 0;
  pre_roll_duration_ = 0.0;
  pre_roll_elapsed_ = 0.0;
  last_pre_roll_log_slot_ = -1;

  start_pub_->on_activate();
  finish_pub_->on_activate();
  RCLCPP_INFO(get_node()->get_logger(),
              "EePoseReplayController activated; HOLD/nullspace will be captured "
              "on the first update tick.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn EePoseReplayController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  start_pub_->on_deactivate();
  finish_pub_->on_deactivate();
  franka_cartesian_pose_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

void EePoseReplayController::modeCallback(const std_msgs::msg::String::SharedPtr msg) {
  std::string mode = msg->data;
  std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);

  if (mode == "hold" || mode == "idle" || mode == "stop") {
    hold_reset_requested_.store(true);
    phase_.store(static_cast<int>(Phase::HOLD));
    RCLCPP_INFO(get_node()->get_logger(), "Phase -> HOLD");
    return;
  }

  if (mode == "replay") {
    if (!has_new_trajectory_.load() && !active_trajectory_) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Refusing REPLAY: no trajectory has been received yet on '%s'",
                  trajectory_topic_.c_str());
      return;
    }
    phase_.store(static_cast<int>(move_to_start_enabled_ ? Phase::PRE_ROLL : Phase::TRACKING));
    RCLCPP_INFO(get_node()->get_logger(), move_to_start_enabled_
                                            ? "Phase -> PRE_ROLL (move-to-start before replay)"
                                            : "Phase -> TRACKING (replay)");
    return;
  }

  RCLCPP_WARN(get_node()->get_logger(), "Unknown mode '%s' (expected hold|replay)",
              msg->data.c_str());
}

void EePoseReplayController::trajectoryCallback(
    const trajectory_msgs::msg::MultiDOFJointTrajectory::SharedPtr msg) {
  if (msg->points.size() < 2) {
    RCLCPP_WARN(get_node()->get_logger(), "Cartesian trajectory has <2 points; ignoring");
    return;
  }

  auto traj = std::make_shared<Trajectory>();
  traj->times.reserve(msg->points.size());
  traj->positions.reserve(msg->points.size());
  traj->orientations.reserve(msg->points.size());
  traj->linear_velocities.reserve(msg->points.size());

  bool any_linear_velocity = false;
  for (size_t i = 0; i < msg->points.size(); ++i) {
    const auto& pt = msg->points[i];
    if (pt.transforms.empty()) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Cartesian trajectory point %zu has no transform; ignoring trajectory", i);
      return;
    }
    const double t = static_cast<double>(pt.time_from_start.sec) +
                     static_cast<double>(pt.time_from_start.nanosec) * 1e-9;
    if (i > 0 && t <= traj->times.back()) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Trajectory time_from_start not strictly increasing at point %zu "
                   "(t=%.6f, prev=%.6f)",
                   i, t, traj->times.back());
      return;
    }

    const auto& transform = pt.transforms.front();
    Eigen::Vector3d position(transform.translation.x, transform.translation.y,
                             transform.translation.z);
    Eigen::Quaterniond orientation =
        normalizedQuaternion(transform.rotation.w, transform.rotation.x, transform.rotation.y,
                             transform.rotation.z);
    if (!traj->orientations.empty() && orientation.dot(traj->orientations.back()) < 0.0) {
      orientation.coeffs() *= -1.0;
    }

    Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero();
    if (!pt.velocities.empty()) {
      linear_velocity = Eigen::Vector3d(pt.velocities.front().linear.x,
                                        pt.velocities.front().linear.y,
                                        pt.velocities.front().linear.z);
      any_linear_velocity = true;
    }

    traj->times.push_back(t);
    traj->positions.push_back(position);
    traj->orientations.push_back(orientation);
    traj->linear_velocities.push_back(linear_velocity);
  }

  const double first_time = traj->times.front();
  for (auto& t : traj->times) {
    t -= first_time;
  }

  if (!any_linear_velocity) {
    const size_t n = traj->times.size();
    for (size_t i = 0; i < n; ++i) {
      double dt = 0.0;
      Eigen::Vector3d dp = Eigen::Vector3d::Zero();
      if (i == 0) {
        dt = traj->times[1] - traj->times[0];
        dp = traj->positions[1] - traj->positions[0];
      } else if (i + 1 == n) {
        dt = traj->times[i] - traj->times[i - 1];
        dp = traj->positions[i] - traj->positions[i - 1];
      } else {
        dt = traj->times[i + 1] - traj->times[i - 1];
        dp = traj->positions[i + 1] - traj->positions[i - 1];
      }
      if (dt > 1e-9) {
        traj->linear_velocities[i] = dp / dt;
      }
    }
  }

  {
    std::lock_guard<std::mutex> lk(trajectory_mutex_);
    next_trajectory_ = traj;
  }
  has_new_trajectory_.store(true);

  RCLCPP_INFO(get_node()->get_logger(),
              "Received EE pose trajectory: %zu points, duration %.3f s",
              traj->times.size(), traj->times.back());
}

void EePoseReplayController::consumePendingTrajectoryIfReady(Phase phase) {
  if (!has_new_trajectory_.load()) {
    return;
  }
  const bool safe_to_swap = phase == Phase::HOLD || !active_trajectory_;
  if (!safe_to_swap) {
    return;
  }
  std::unique_lock<std::mutex> lk(trajectory_mutex_, std::try_to_lock);
  if (lk.owns_lock() && next_trajectory_) {
    active_trajectory_ = next_trajectory_;
    next_trajectory_.reset();
    has_new_trajectory_.store(false);
    tracking_initialized_ = false;
    replay_elapsed_ = 0.0;
    segment_index_ = 0;
    pre_roll_duration_ = 0.0;
    pre_roll_elapsed_ = 0.0;
  }
}

void EePoseReplayController::updateJointStates() {
  for (int i = 0; i < kNumJoints; ++i) {
    q_(i) = state_interfaces_[2 * i].get_value();
    dq_(i) = state_interfaces_[2 * i + 1].get_value();
  }
  constexpr double kAlpha = 0.99;
  dq_filtered_ = (1.0 - kAlpha) * dq_filtered_ + kAlpha * dq_;
}

void EePoseReplayController::updateCurrentPose() {
  Eigen::Quaterniond q;
  std::tie(q, current_position_) =
      franka_cartesian_pose_->getCurrentOrientationAndTranslation();
  if (q.norm() <= 1e-9) {
    current_orientation_ = Eigen::Quaterniond::Identity();
  } else {
    current_orientation_ = q.normalized();
  }
}

bool EePoseReplayController::computeDesiredPose(const rclcpp::Duration& period) {
  Phase phase = static_cast<Phase>(phase_.load());
  if (hold_reset_requested_.exchange(false)) {
    hold_initialized_ = false;
  }
  consumePendingTrajectoryIfReady(phase);

  if (phase == Phase::PRE_ROLL && active_trajectory_) {
    const auto& traj = *active_trajectory_;
    if (pre_roll_duration_ <= 0.0) {
      if (current_position_.norm() < 1e-3) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "PRE_ROLL aborted: current_position_ looks uninitialized "
                    "(%.6f,%.6f,%.6f). Falling back to HOLD; try replay again.",
                    current_position_.x(), current_position_.y(),
                    current_position_.z());
        phase_.store(static_cast<int>(Phase::HOLD));
        hold_initialized_ = false;
        pre_roll_duration_ = 0.0;
        pre_roll_elapsed_ = 0.0;
        return false;
      }

      pre_roll_position_start_ = last_command_position_;
      pre_roll_orientation_start_ = last_command_orientation_;
      pre_roll_elapsed_ = 0.0;
      last_pre_roll_log_slot_ = -1;

      const Eigen::Vector3d target_position = traj.positions.front();
      const Eigen::Quaterniond target_orientation = traj.orientations.front();
      const double translation_delta = (target_position - pre_roll_position_start_).norm();
      const double rotation_delta =
          pre_roll_orientation_start_.angularDistance(target_orientation);
      constexpr double kMinJerkPeakFactor = 1.875;
      pre_roll_duration_ =
          std::max({move_to_start_min_duration_,
                    kMinJerkPeakFactor * translation_delta /
                        move_to_start_max_translation_velocity_,
                    kMinJerkPeakFactor * rotation_delta /
                        move_to_start_max_rotation_velocity_});
      tracking_initialized_ = false;
      RCLCPP_INFO(get_node()->get_logger(),
                  "EE pre-roll: translation %.4f m, rotation %.4f rad, duration %.2f s",
                  translation_delta, rotation_delta, pre_roll_duration_);
      RCLCPP_INFO(
          get_node()->get_logger(),
          "  start    pos=(%.4f,%.4f,%.4f) target pos=(%.4f,%.4f,%.4f)",
          pre_roll_position_start_.x(), pre_roll_position_start_.y(),
          pre_roll_position_start_.z(), target_position.x(), target_position.y(),
          target_position.z());
      RCLCPP_INFO(get_node()->get_logger(),
                  "  current vs last_cmd drift=%.6f m",
                  (current_position_ - last_command_position_).norm());
    }

    pre_roll_elapsed_ += period.seconds();
    const double s = minJerkPositionScale(pre_roll_elapsed_, pre_roll_duration_);
    desired_position_ =
        pre_roll_position_start_ + s * (traj.positions.front() - pre_roll_position_start_);
    desired_orientation_ = pre_roll_orientation_start_.slerp(s, traj.orientations.front());

    const int slot = static_cast<int>(pre_roll_elapsed_ / 0.5);
    if (slot != last_pre_roll_log_slot_) {
      last_pre_roll_log_slot_ = slot;
      RCLCPP_INFO(get_node()->get_logger(),
                  "PRE_ROLL t=%.2f/%.2f s=%.3f cmd=(%.3f,%.3f,%.3f) cur=(%.3f,%.3f,%.3f) "
                  "|err|=%.4f",
                  pre_roll_elapsed_, pre_roll_duration_, s,
                  desired_position_.x(), desired_position_.y(), desired_position_.z(),
                  current_position_.x(), current_position_.y(), current_position_.z(),
                  (desired_position_ - current_position_).norm());
    }

    if (pre_roll_elapsed_ >= pre_roll_duration_) {
      phase_.store(static_cast<int>(Phase::TRACKING));
      pre_roll_duration_ = 0.0;
      pre_roll_elapsed_ = 0.0;
      tracking_initialized_ = false;
      RCLCPP_INFO(get_node()->get_logger(),
                  "EE pre-roll complete -> TRACKING; cmd=(%.4f,%.4f,%.4f) cur=(%.4f,%.4f,%.4f)",
                  desired_position_.x(), desired_position_.y(), desired_position_.z(),
                  current_position_.x(), current_position_.y(), current_position_.z());
    }
    return true;
  }

  if (phase == Phase::TRACKING && active_trajectory_) {
    const auto& traj = *active_trajectory_;
    if (!tracking_initialized_) {
      replay_elapsed_ = 0.0;
      segment_index_ = 0;
      tracking_initialized_ = true;
      start_pending_ = true;
      RCLCPP_INFO(get_node()->get_logger(),
                  "TRACKING start: cur=(%.4f,%.4f,%.4f) traj[0]=(%.4f,%.4f,%.4f) duration=%.2fs",
                  current_position_.x(), current_position_.y(), current_position_.z(),
                  traj.positions.front().x(), traj.positions.front().y(),
                  traj.positions.front().z(), traj.times.back());
    }

    replay_elapsed_ += period.seconds();
    const double t_end = traj.times.back();
    if (replay_elapsed_ >= t_end) {
      desired_position_ = traj.positions.back();
      desired_orientation_ = traj.orientations.back();
      hold_position_ = desired_position_;
      hold_orientation_ = desired_orientation_;
      hold_initialized_ = true;
      finish_pending_ = true;
      phase_.store(static_cast<int>(Phase::HOLD));
      tracking_initialized_ = false;
      RCLCPP_INFO(get_node()->get_logger(),
                  "EE replay finished; cmd=(%.4f,%.4f,%.4f) cur=(%.4f,%.4f,%.4f)",
                  desired_position_.x(), desired_position_.y(), desired_position_.z(),
                  current_position_.x(), current_position_.y(), current_position_.z());
      return true;
    }

    while (segment_index_ + 1 < traj.times.size() &&
           traj.times[segment_index_ + 1] <= replay_elapsed_) {
      ++segment_index_;
    }
    const size_t i = std::min(segment_index_, traj.times.size() - 2);
    const double t0 = traj.times[i];
    const double t1 = traj.times[i + 1];
    const double h = std::max(1e-9, t1 - t0);
    const double s = std::clamp((replay_elapsed_ - t0) / h, 0.0, 1.0);
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    const double h10 = s3 - 2.0 * s2 + s;
    const double h01 = -2.0 * s3 + 3.0 * s2;
    const double h11 = s3 - s2;
    desired_position_ = h00 * traj.positions[i] + h10 * h * traj.linear_velocities[i] +
                        h01 * traj.positions[i + 1] +
                        h11 * h * traj.linear_velocities[i + 1];
    desired_orientation_ = traj.orientations[i].slerp(s, traj.orientations[i + 1]);
    return true;
  }

  // HOLD branch
  if (!hold_initialized_) {
    hold_position_ = current_position_;
    hold_orientation_ = current_orientation_;
    hold_initialized_ = true;
    nullspace_q_target_ = q_;
    RCLCPP_INFO(
        get_node()->get_logger(),
        "HOLD initialized: pos=(%.4f,%.4f,%.4f) q=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]",
        hold_position_.x(), hold_position_.y(), hold_position_.z(),
        q_(0), q_(1), q_(2), q_(3), q_(4), q_(5), q_(6));
  }
  desired_position_ = hold_position_;
  desired_orientation_ = hold_orientation_;
  pre_roll_duration_ = 0.0;
  pre_roll_elapsed_ = 0.0;
  replay_elapsed_ = 0.0;
  segment_index_ = 0;
  tracking_initialized_ = false;
  return true;
}

EePoseReplayController::Vector7d EePoseReplayController::computeImpedanceTorque(
    const Matrix6x7d& jacobian) {
  // Pose error in base frame: [position, axis-angle].
  Eigen::Matrix<double, 6, 1> error;
  error.head<3>() = current_position_ - desired_position_;

  Eigen::Quaterniond orientation_current = current_orientation_;
  if (desired_orientation_.coeffs().dot(orientation_current.coeffs()) < 0.0) {
    orientation_current.coeffs() << -orientation_current.coeffs();
  }
  Eigen::Quaterniond error_quat(orientation_current.inverse() * desired_orientation_);
  Eigen::Vector3d orientation_error_local(error_quat.x(), error_quat.y(), error_quat.z());
  error.tail<3>() = -orientation_current.toRotationMatrix() * orientation_error_local;

  // Task-space impedance: tau_task = J^T * (-K*err - D*(J*dq))
  Vector7d tau_task =
      jacobian.transpose() *
      (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq_filtered_));

  // Nullspace torque pulling q toward nullspace_q_target_.
  const double nullspace_damping = 2.0 * std::sqrt(std::max(nullspace_stiffness_, 0.0));
  Eigen::Matrix<double, 7, 6> J_pinv = dampedPseudoInverse(jacobian);
  Eigen::Matrix<double, 7, 7> N =
      Eigen::Matrix<double, 7, 7>::Identity() - jacobian.transpose() * J_pinv.transpose();
  Vector7d tau_null = N * (nullspace_stiffness_ * (nullspace_q_target_ - q_) -
                           nullspace_damping * dq_filtered_);

  return tau_task + tau_null;
}

controller_interface::return_type EePoseReplayController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  updateJointStates();
  updateCurrentPose();

  if (needs_initialization_) {
    if (current_position_.norm() < 1e-3) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                           "Waiting for valid EE pose (current_position_=(%.4f,%.4f,%.4f))",
                           current_position_.x(), current_position_.y(),
                           current_position_.z());
      for (int i = 0; i < kNumJoints; ++i) {
        command_interfaces_[i].set_value(0.0);
      }
      return controller_interface::return_type::OK;
    }
    hold_position_ = current_position_;
    hold_orientation_ = current_orientation_;
    hold_initialized_ = true;
    last_command_position_ = current_position_;
    last_command_orientation_ = current_orientation_;
    last_command_initialized_ = true;
    nullspace_q_target_ = q_;
    needs_initialization_ = false;
    RCLCPP_INFO(get_node()->get_logger(),
                "Init complete: HOLD pos=(%.4f,%.4f,%.4f) q=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]",
                hold_position_.x(), hold_position_.y(), hold_position_.z(),
                q_(0), q_(1), q_(2), q_(3), q_(4), q_(5), q_(6));
  }

  if (!computeDesiredPose(period)) {
    desired_position_ = current_position_;
    desired_orientation_ = current_orientation_;
  }

  Matrix6x7d jacobian = computeJacobian();
  Vector7d tau = computeImpedanceTorque(jacobian);

  // Sparse diagnostic: ~1 Hz, prints whether things look sane.
  static int diag_counter = 0;
  if (++diag_counter >= 1000) {
    diag_counter = 0;
    Eigen::Vector3d pos_err = current_position_ - desired_position_;
    double j_v_norm = jacobian.topRows<3>().norm();
    double j_w_norm = jacobian.bottomRows<3>().norm();
    Phase phase = static_cast<Phase>(phase_.load());
    const char* phase_str = phase == Phase::HOLD       ? "HOLD"
                            : phase == Phase::PRE_ROLL ? "PRE_ROLL"
                                                       : "TRACKING";
    RCLCPP_INFO(get_node()->get_logger(),
                "DIAG phase=%s |pos_err|=%.4fm |J_v|=%.2f |J_w|=%.2f "
                "tau=[%.2f %.2f %.2f %.2f %.2f %.2f %.2f]",
                phase_str, pos_err.norm(), j_v_norm, j_w_norm,
                tau(0), tau(1), tau(2), tau(3), tau(4), tau(5), tau(6));
  }

  for (int i = 0; i < kNumJoints; ++i) {
    command_interfaces_[i].set_value(tau(i));
  }

  last_command_position_ = desired_position_;
  last_command_orientation_ = desired_orientation_;
  last_command_initialized_ = true;

  if (start_pending_) {
    publishStarted();
    start_pending_ = false;
  }
  if (finish_pending_) {
    publishFinished();
    finish_pending_ = false;
  }
  return controller_interface::return_type::OK;
}

void EePoseReplayController::publishStarted() {
  std_msgs::msg::Bool msg;
  msg.data = true;
  start_pub_->publish(msg);
}

void EePoseReplayController::publishFinished() {
  std_msgs::msg::Bool msg;
  msg.data = true;
  finish_pub_->publish(msg);
}

}  // namespace franka_example_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::EePoseReplayController,
                       controller_interface::ControllerInterface)
