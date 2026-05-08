// Copyright (c) 2026 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <franka_example_controllers/teach_replay_controller.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <string>

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
TeachReplayController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
TeachReplayController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

CallbackReturn TeachReplayController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<std::string>("mode_topic", "/teach_replay/mode");
    auto_declare<std::string>("trajectory_topic", "/teach_replay/trajectory");
    auto_declare<std::string>("start_topic", "/teach_replay/replay_started");
    auto_declare<std::string>("finish_topic", "/teach_replay/replay_finished");
  } catch (const std::exception& e) {
    fprintf(stderr, "TeachReplayController init failed: %s\n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn TeachReplayController::on_configure(const rclcpp_lifecycle::State& /*prev*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  mode_topic_ = get_node()->get_parameter("mode_topic").as_string();
  trajectory_topic_ = get_node()->get_parameter("trajectory_topic").as_string();
  start_topic_ = get_node()->get_parameter("start_topic").as_string();
  finish_topic_ = get_node()->get_parameter("finish_topic").as_string();

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  if (k_gains.size() != static_cast<size_t>(kNumJoints) ||
      d_gains.size() != static_cast<size_t>(kNumJoints)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "k_gains/d_gains must have %d elements (got %zu / %zu)",
                 kNumJoints, k_gains.size(), d_gains.size());
    return CallbackReturn::FAILURE;
  }
  for (int i = 0; i < kNumJoints; ++i) {
    k_gains_(i) = k_gains[i];
    d_gains_(i) = d_gains[i];
  }
  dq_filtered_.setZero();

  mode_sub_ = get_node()->create_subscription<std_msgs::msg::String>(
      mode_topic_, rclcpp::QoS(10),
      std::bind(&TeachReplayController::modeCallback, this, std::placeholders::_1));

  trajectory_sub_ = get_node()->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      trajectory_topic_, rclcpp::QoS(1),
      std::bind(&TeachReplayController::trajectoryCallback, this, std::placeholders::_1));

  start_pub_ = get_node()->create_publisher<std_msgs::msg::Bool>(start_topic_, rclcpp::QoS(10));
  finish_pub_ = get_node()->create_publisher<std_msgs::msg::Bool>(finish_topic_, rclcpp::QoS(10));

  RCLCPP_INFO(get_node()->get_logger(),
              "TeachReplayController configured. mode_topic=%s trajectory_topic=%s",
              mode_topic_.c_str(), trajectory_topic_.c_str());
  return CallbackReturn::SUCCESS;
}

CallbackReturn TeachReplayController::on_activate(const rclcpp_lifecycle::State& /*prev*/) {
  updateJointStates();
  dq_filtered_.setZero();
  hold_q_ = q_;
  mode_.store(static_cast<int>(Mode::TEACH));
  replay_running_ = false;
  replay_elapsed_ = 0.0;
  active_trajectory_.reset();
  has_new_trajectory_.store(false);
  start_pub_->on_activate();
  finish_pub_->on_activate();
  RCLCPP_INFO(get_node()->get_logger(), "TeachReplayController activated in TEACH mode");
  return CallbackReturn::SUCCESS;
}

CallbackReturn TeachReplayController::on_deactivate(const rclcpp_lifecycle::State& /*prev*/) {
  start_pub_->on_deactivate();
  finish_pub_->on_deactivate();
  return CallbackReturn::SUCCESS;
}

void TeachReplayController::modeCallback(const std_msgs::msg::String::SharedPtr msg) {
  std::string m = msg->data;
  std::transform(m.begin(), m.end(), m.begin(), ::tolower);
  if (m == "teach" || m == "idle") {
    mode_.store(static_cast<int>(Mode::TEACH));
    RCLCPP_INFO(get_node()->get_logger(), "Mode -> TEACH (zero torque)");
  } else if (m == "replay") {
    if (!has_new_trajectory_.load() && !active_trajectory_) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Refusing REPLAY: no trajectory has been received yet on '%s'",
                  trajectory_topic_.c_str());
      return;
    }
    mode_.store(static_cast<int>(Mode::REPLAY));
    RCLCPP_INFO(get_node()->get_logger(), "Mode -> REPLAY");
  } else {
    RCLCPP_WARN(get_node()->get_logger(), "Unknown mode '%s' (expected teach|replay)",
                msg->data.c_str());
  }
}

void TeachReplayController::trajectoryCallback(
    const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
  if (msg->points.size() < 2) {
    RCLCPP_WARN(get_node()->get_logger(), "Trajectory has <2 points; ignoring");
    return;
  }

  auto traj = std::make_shared<Trajectory>();
  traj->times.reserve(msg->points.size());
  traj->positions.reserve(msg->points.size());
  traj->velocities.reserve(msg->points.size());

  for (size_t i = 0; i < msg->points.size(); ++i) {
    const auto& pt = msg->points[i];
    if (pt.positions.size() < static_cast<size_t>(kNumJoints)) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Trajectory point %zu has %zu positions (need >= %d); ignoring trajectory",
                   i, pt.positions.size(), kNumJoints);
      return;
    }
    double t = static_cast<double>(pt.time_from_start.sec) +
               static_cast<double>(pt.time_from_start.nanosec) * 1e-9;
    if (i > 0 && t <= traj->times.back()) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Trajectory time_from_start not strictly increasing at point %zu (t=%.4f, prev=%.4f)",
                   i, t, traj->times.back());
      return;
    }
    Vector7d q;
    for (int j = 0; j < kNumJoints; ++j) {
      q(j) = pt.positions[j];
    }
    Vector7d v = Vector7d::Zero();
    if (pt.velocities.size() >= static_cast<size_t>(kNumJoints)) {
      for (int j = 0; j < kNumJoints; ++j) {
        v(j) = pt.velocities[j];
      }
    }
    traj->times.push_back(t);
    traj->positions.push_back(q);
    traj->velocities.push_back(v);
  }

  // Fill in missing velocities with central differences.
  bool any_velocity = false;
  for (const auto& pt : msg->points) {
    if (pt.velocities.size() >= static_cast<size_t>(kNumJoints)) {
      any_velocity = true;
      break;
    }
  }
  if (!any_velocity) {
    const size_t n = traj->times.size();
    for (size_t i = 0; i < n; ++i) {
      if (i == 0) {
        double dt = traj->times[1] - traj->times[0];
        traj->velocities[i] = dt > 1e-9 ? (traj->positions[1] - traj->positions[0]) / dt
                                        : Vector7d::Zero();
      } else if (i + 1 == n) {
        double dt = traj->times[i] - traj->times[i - 1];
        traj->velocities[i] = dt > 1e-9 ? (traj->positions[i] - traj->positions[i - 1]) / dt
                                        : Vector7d::Zero();
      } else {
        double dt = traj->times[i + 1] - traj->times[i - 1];
        traj->velocities[i] = dt > 1e-9
                                  ? (traj->positions[i + 1] - traj->positions[i - 1]) / dt
                                  : Vector7d::Zero();
      }
    }
  }

  {
    std::lock_guard<std::mutex> lk(trajectory_mutex_);
    next_trajectory_ = traj;
  }
  has_new_trajectory_.store(true);

  RCLCPP_INFO(get_node()->get_logger(),
              "Received trajectory: %zu points, duration %.3f s",
              traj->times.size(), traj->times.back());
}

void TeachReplayController::updateJointStates() {
  for (int i = 0; i < kNumJoints; ++i) {
    const auto& pos_iface = state_interfaces_.at(2 * i);
    const auto& vel_iface = state_interfaces_.at(2 * i + 1);
    assert(pos_iface.get_interface_name() == "position");
    assert(vel_iface.get_interface_name() == "velocity");
    q_(i) = pos_iface.get_value();
    dq_(i) = vel_iface.get_value();
  }
}

void TeachReplayController::publishStarted() {
  std_msgs::msg::Bool msg;
  msg.data = true;
  start_pub_->publish(msg);
}

void TeachReplayController::publishFinished() {
  std_msgs::msg::Bool msg;
  msg.data = true;
  finish_pub_->publish(msg);
}

controller_interface::return_type TeachReplayController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  updateJointStates();

  // Low-pass filter velocity estimate (matches joint_impedance_example_controller).
  constexpr double kAlpha = 0.99;
  dq_filtered_ = (1.0 - kAlpha) * dq_filtered_ + kAlpha * dq_;

  // Pull pending trajectory in non-blocking fashion.
  if (has_new_trajectory_.load()) {
    std::unique_lock<std::mutex> lk(trajectory_mutex_, std::try_to_lock);
    if (lk.owns_lock() && next_trajectory_) {
      active_trajectory_ = next_trajectory_;
      next_trajectory_.reset();
      has_new_trajectory_.store(false);
      replay_running_ = false;  // require explicit REPLAY mode to (re)start
      replay_elapsed_ = 0.0;
    }
  }

  Vector7d q_d = q_;
  Vector7d dq_d = Vector7d::Zero();

  const auto current_mode = static_cast<Mode>(mode_.load());

  if (current_mode == Mode::REPLAY && active_trajectory_) {
    if (!replay_running_) {
      replay_running_ = true;
      replay_elapsed_ = 0.0;
      start_pending_ = true;
      // Snap reference to current pose at start to avoid jolt; first segment will
      // smoothly re-converge to traj[0] -> traj[1] interpolation.
    }

    replay_elapsed_ += period.seconds();
    const auto& traj = *active_trajectory_;
    const double t_end = traj.times.back();

    if (replay_elapsed_ >= t_end) {
      // Hold last point.
      q_d = traj.positions.back();
      dq_d.setZero();
      hold_q_ = q_d;
      if (replay_running_) {
        replay_running_ = false;
        finish_pending_ = true;
        // Auto-revert to TEACH so the user can drag again without sending a mode msg.
        mode_.store(static_cast<int>(Mode::TEACH));
        RCLCPP_INFO(get_node()->get_logger(), "Replay finished; reverted to TEACH");
      }
    } else {
      // Find segment [i, i+1] with traj.times[i] <= t < traj.times[i+1].
      // Linear scan is fine: trajectories are at most a few thousand points.
      size_t i = 0;
      while (i + 1 < traj.times.size() && traj.times[i + 1] <= replay_elapsed_) {
        ++i;
      }
      double t0 = traj.times[i];
      double t1 = traj.times[i + 1];
      double s = (replay_elapsed_ - t0) / std::max(1e-9, (t1 - t0));
      q_d = traj.positions[i] + s * (traj.positions[i + 1] - traj.positions[i]);
      dq_d = traj.velocities[i] + s * (traj.velocities[i + 1] - traj.velocities[i]);
    }

    // Joint-impedance torque with velocity feedforward.
    Vector7d tau =
        k_gains_.cwiseProduct(q_d - q_) + d_gains_.cwiseProduct(dq_d - dq_filtered_);
    for (int i2 = 0; i2 < kNumJoints; ++i2) {
      command_interfaces_[i2].set_value(tau(i2));
    }
  } else {
    // TEACH (or REPLAY without a trajectory): zero torque, gravity compensation
    // is handled internally by the Franka hardware.
    for (auto& cmd : command_interfaces_) {
      cmd.set_value(0.0);
    }
    // Track current pose so a subsequent REPLAY hold reference is sensible.
    hold_q_ = q_;
    if (replay_running_) {
      // User flipped mode away mid-replay; cancel cleanly.
      replay_running_ = false;
      finish_pending_ = true;
      RCLCPP_INFO(get_node()->get_logger(), "Replay aborted; back to TEACH");
    }
  }

  // Publish edge-triggered notifications outside the inner branches so they
  // fire exactly once per transition regardless of which branch ran.
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
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::TeachReplayController,
                       controller_interface::ControllerInterface)
