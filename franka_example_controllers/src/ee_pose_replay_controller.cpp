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

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cmath>
#include <exception>
#include <functional>
#include <future>
#include <string>

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

std::array<double, 16> makeColumnMajorPose(const Eigen::Quaterniond& orientation,
                                           const Eigen::Vector3d& position) {
  Eigen::Quaterniond normalized_orientation = orientation.normalized();
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  pose.block<3, 3>(0, 0) = normalized_orientation.toRotationMatrix();
  pose.block<3, 1>(0, 3) = position;

  std::array<double, 16> out{};
  std::copy(pose.data(), pose.data() + out.size(), out.begin());
  return out;
}

Eigen::Quaterniond normalizedQuaternion(double w, double x, double y, double z) {
  Eigen::Quaterniond q(w, x, y, z);
  if (q.norm() <= 1e-9) {
    return Eigen::Quaterniond::Identity();
  }
  q.normalize();
  return q;
}

}  // namespace

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
EePoseReplayController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_pose_->get_command_interface_names();
  return config;
}

controller_interface::InterfaceConfiguration
EePoseReplayController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_pose_->get_state_interface_names();
  return config;
}

CallbackReturn EePoseReplayController::on_init() {
  try {
    auto_declare<bool>("gazebo", false);
    auto_declare<std::string>("mode_topic", "/ee_pose_replay/mode");
    auto_declare<std::string>("trajectory_topic", "/ee_pose_replay/trajectory");
    auto_declare<std::string>("start_topic", "/ee_pose_replay/replay_started");
    auto_declare<std::string>("finish_topic", "/ee_pose_replay/replay_finished");
    auto_declare<bool>("move_to_start", true);
    auto_declare<double>("move_to_start_min_duration", 4.0);
    auto_declare<double>("move_to_start_max_translation_velocity", 0.05);
    auto_declare<double>("move_to_start_max_rotation_velocity", 0.5);
  } catch (const std::exception& e) {
    fprintf(stderr, "EePoseReplayController init failed: %s\n", e.what());
    return CallbackReturn::ERROR;
  }

  franka_cartesian_pose_ =
      std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(
          k_elbow_activated_);
  return CallbackReturn::SUCCESS;
}

CallbackReturn EePoseReplayController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
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

  if (move_to_start_max_translation_velocity_ <= 0.0 ||
      move_to_start_max_rotation_velocity_ <= 0.0) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "move_to_start max translation/rotation velocities must be > 0");
    return CallbackReturn::FAILURE;
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
          std::bind(&EePoseReplayController::trajectoryCallback, this,
                    std::placeholders::_1));
  start_pub_ = get_node()->create_publisher<std_msgs::msg::Bool>(start_topic_, rclcpp::QoS(10));
  finish_pub_ = get_node()->create_publisher<std_msgs::msg::Bool>(finish_topic_, rclcpp::QoS(10));

  RCLCPP_INFO(get_node()->get_logger(),
              "EePoseReplayController configured. mode_topic=%s trajectory_topic=%s",
              mode_topic_.c_str(), trajectory_topic_.c_str());
  return CallbackReturn::SUCCESS;
}

CallbackReturn EePoseReplayController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_pose_->assign_loaned_command_interfaces(command_interfaces_);
  franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_);
  hold_initialized_ = false;
  phase_.store(static_cast<int>(Phase::HOLD));
  active_trajectory_.reset();
  next_trajectory_.reset();
  has_new_trajectory_.store(false);
  start_pending_ = false;
  finish_pending_ = false;
  tracking_initialized_ = false;
  hold_reset_requested_.store(false);
  last_command_initialized_ = false;
  replay_elapsed_ = 0.0;
  segment_index_ = 0;
  pre_roll_duration_ = 0.0;
  pre_roll_elapsed_ = 0.0;
  start_pub_->on_activate();
  finish_pub_->on_activate();
  RCLCPP_INFO(get_node()->get_logger(), "EePoseReplayController activated in HOLD mode");
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

void EePoseReplayController::updateCurrentPose() {
  std::tie(current_orientation_, current_position_) =
      franka_cartesian_pose_->getCurrentOrientationAndTranslation();
  if (current_orientation_.norm() <= 1e-9) {
    current_orientation_ = Eigen::Quaterniond::Identity();
  } else {
    current_orientation_.normalize();
  }
}

bool EePoseReplayController::commandPose(const Eigen::Quaterniond& orientation,
                                         const Eigen::Vector3d& position) {
  const Eigen::Quaterniond normalized_orientation = orientation.normalized();
  const auto pose_command = makeColumnMajorPose(normalized_orientation, position);
  if (franka_cartesian_pose_->setCommand(pose_command)) {
    last_command_orientation_ = normalized_orientation;
    last_command_position_ = position;
    last_command_initialized_ = true;
    return true;
  }
  RCLCPP_FATAL(get_node()->get_logger(), "Set Cartesian pose command failed.");
  return false;
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

controller_interface::return_type EePoseReplayController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  updateCurrentPose();
  Phase phase = static_cast<Phase>(phase_.load());
  if (hold_reset_requested_.exchange(false)) {
    hold_initialized_ = false;
  }
  consumePendingTrajectoryIfReady(phase);

  Eigen::Vector3d position_command = current_position_;
  Eigen::Quaterniond orientation_command = current_orientation_;

  if (phase == Phase::PRE_ROLL && active_trajectory_) {
    const auto& traj = *active_trajectory_;
    if (pre_roll_duration_ <= 0.0) {
      // Sanity check: refuse pre-roll if current pose looks like an uninitialized
      // default (close to origin with identity orientation), which usually means
      // the state interface hasn't been read yet. Fall back to HOLD so libfranka
      // doesn't get fed a 50 cm command jump.
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
        return controller_interface::return_type::OK;
      }

      pre_roll_position_start_ = current_position_;
      pre_roll_orientation_start_ = current_orientation_;
      pre_roll_elapsed_ = 0.0;

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
          "  start    pos=(%.4f,%.4f,%.4f) quat=(%.4f,%.4f,%.4f,%.4f)",
          pre_roll_position_start_.x(), pre_roll_position_start_.y(),
          pre_roll_position_start_.z(), pre_roll_orientation_start_.x(),
          pre_roll_orientation_start_.y(), pre_roll_orientation_start_.z(),
          pre_roll_orientation_start_.w());
      RCLCPP_INFO(
          get_node()->get_logger(),
          "  target   pos=(%.4f,%.4f,%.4f) quat=(%.4f,%.4f,%.4f,%.4f)",
          target_position.x(), target_position.y(), target_position.z(),
          target_orientation.x(), target_orientation.y(), target_orientation.z(),
          target_orientation.w());
      const Eigen::Vector3d last_cmd_drift =
          last_command_initialized_
              ? (current_position_ - last_command_position_)
              : Eigen::Vector3d::Zero();
      RCLCPP_INFO(get_node()->get_logger(),
                  "  current_vs_last_cmd drift=%.6f m (last_cmd_initialized=%d)",
                  last_cmd_drift.norm(), last_command_initialized_ ? 1 : 0);
    }

    pre_roll_elapsed_ += period.seconds();
    const double s = minJerkPositionScale(pre_roll_elapsed_, pre_roll_duration_);
    position_command =
        pre_roll_position_start_ + s * (traj.positions.front() - pre_roll_position_start_);
    orientation_command = pre_roll_orientation_start_.slerp(s, traj.orientations.front());

    // First few ticks of pre-roll: print the commanded pose so we can see
    // whether libfranka is being fed a jump.
    if (pre_roll_elapsed_ <= 3.5 * period.seconds()) {
      RCLCPP_INFO(get_node()->get_logger(),
                  "PRE_ROLL tick t=%.4f s=%.3e cmd_pos=(%.4f,%.4f,%.4f)",
                  pre_roll_elapsed_, s, position_command.x(), position_command.y(),
                  position_command.z());
    }

    if (pre_roll_elapsed_ >= pre_roll_duration_) {
      phase_.store(static_cast<int>(Phase::TRACKING));
      pre_roll_duration_ = 0.0;
      pre_roll_elapsed_ = 0.0;
      tracking_initialized_ = false;
      RCLCPP_INFO(get_node()->get_logger(),
                  "EE pre-roll complete -> TRACKING; last cmd=(%.4f,%.4f,%.4f)",
                  position_command.x(), position_command.y(), position_command.z());
    }
  } else if (phase == Phase::TRACKING && active_trajectory_) {
    const auto& traj = *active_trajectory_;
    if (!tracking_initialized_) {
      replay_elapsed_ = 0.0;
      segment_index_ = 0;
      tracking_initialized_ = true;
      start_pending_ = true;
      const Eigen::Vector3d traj_v0 = traj.linear_velocities.front();
      RCLCPP_INFO(
          get_node()->get_logger(),
          "TRACKING start: cur=(%.4f,%.4f,%.4f) traj[0]=(%.4f,%.4f,%.4f) "
          "traj_v0=(%.4f,%.4f,%.4f) |v0|=%.4f m/s",
          current_position_.x(), current_position_.y(), current_position_.z(),
          traj.positions.front().x(), traj.positions.front().y(),
          traj.positions.front().z(), traj_v0.x(), traj_v0.y(), traj_v0.z(),
          traj_v0.norm());
    }

    replay_elapsed_ += period.seconds();
    const double t_end = traj.times.back();
    if (replay_elapsed_ >= t_end) {
      position_command = traj.positions.back();
      orientation_command = traj.orientations.back();
      hold_position_ = position_command;
      hold_orientation_ = orientation_command;
      hold_initialized_ = true;
      finish_pending_ = true;
      phase_.store(static_cast<int>(Phase::HOLD));
      tracking_initialized_ = false;
      RCLCPP_INFO(get_node()->get_logger(), "EE replay finished; reverted to HOLD");
    } else {
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

      position_command = h00 * traj.positions[i] + h10 * h * traj.linear_velocities[i] +
                         h01 * traj.positions[i + 1] +
                         h11 * h * traj.linear_velocities[i + 1];
      orientation_command = traj.orientations[i].slerp(s, traj.orientations[i + 1]);
    }
  } else {
    if (!hold_initialized_) {
      hold_position_ = current_position_;
      hold_orientation_ = current_orientation_;
      hold_initialized_ = true;
      RCLCPP_INFO(
          get_node()->get_logger(),
          "HOLD initialized: pos=(%.4f,%.4f,%.4f) quat=(%.4f,%.4f,%.4f,%.4f) "
          "|pos|=%.4f",
          hold_position_.x(), hold_position_.y(), hold_position_.z(),
          hold_orientation_.x(), hold_orientation_.y(), hold_orientation_.z(),
          hold_orientation_.w(), hold_position_.norm());
    }
    position_command = hold_position_;
    orientation_command = hold_orientation_;
    pre_roll_duration_ = 0.0;
    pre_roll_elapsed_ = 0.0;
    replay_elapsed_ = 0.0;
    segment_index_ = 0;
    tracking_initialized_ = false;
  }

  if (!commandPose(orientation_command, position_command)) {
    return controller_interface::return_type::ERROR;
  }

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

}  // namespace franka_example_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::EePoseReplayController,
                       controller_interface::ControllerInterface)
