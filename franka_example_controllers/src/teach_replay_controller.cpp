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
    auto_declare<bool>("move_to_start", true);
    auto_declare<double>("move_to_start_min_duration", 4.0);
    auto_declare<double>("move_to_start_max_velocity", 0.2);
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
  move_to_start_enabled_ = get_node()->get_parameter("move_to_start").as_bool();
  move_to_start_min_duration_ =
      get_node()->get_parameter("move_to_start_min_duration").as_double();
  move_to_start_max_velocity_ =
      get_node()->get_parameter("move_to_start_max_velocity").as_double();
  if (move_to_start_max_velocity_ <= 0.0) {
    RCLCPP_FATAL(get_node()->get_logger(), "move_to_start_max_velocity must be > 0");
    return CallbackReturn::FAILURE;
  }

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
  phase_.store(static_cast<int>(Phase::TEACH));
  replay_elapsed_ = 0.0;
  pre_roll_elapsed_ = 0.0;
  pre_roll_duration_ = 0.0;
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
    phase_.store(static_cast<int>(Phase::TEACH));
    RCLCPP_INFO(get_node()->get_logger(), "Phase -> TEACH (zero torque)");
  } else if (m == "replay") {
    if (!has_new_trajectory_.load() && !active_trajectory_) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "Refusing REPLAY: no trajectory has been received yet on '%s'",
                  trajectory_topic_.c_str());
      return;
    }
    // Choose entry phase: if move_to_start is enabled, run a min-jerk move
    // from the current pose to the trajectory's first point first, then
    // start TRACKING. Otherwise jump straight to TRACKING.
    if (move_to_start_enabled_) {
      phase_.store(static_cast<int>(Phase::PRE_ROLL));
      RCLCPP_INFO(get_node()->get_logger(),
                  "Phase -> PRE_ROLL (move-to-start before replay)");
    } else {
      phase_.store(static_cast<int>(Phase::TRACKING));
      RCLCPP_INFO(get_node()->get_logger(), "Phase -> TRACKING (replay)");
    }
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
      double dt;
      Vector7d dq;
      if (i == 0) {
        dt = traj->times[1] - traj->times[0];
        dq = traj->positions[1] - traj->positions[0];
      } else if (i + 1 == n) {
        dt = traj->times[i] - traj->times[i - 1];
        dq = traj->positions[i] - traj->positions[i - 1];
      } else {
        dt = traj->times[i + 1] - traj->times[i - 1];
        dq = traj->positions[i + 1] - traj->positions[i - 1];
      }
      if (dt > 1e-9) {
        traj->velocities[i] = dq / dt;
      } else {
        traj->velocities[i].setZero();
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

// Quintic minimum-jerk interpolation along [0, T] from q0 to qf.
// Returns position and velocity at time t (clamped to [0, T]).
static void minJerk(const Eigen::Matrix<double, 7, 1>& q0,
                    const Eigen::Matrix<double, 7, 1>& qf,
                    double t, double T,
                    Eigen::Matrix<double, 7, 1>& q_out,
                    Eigen::Matrix<double, 7, 1>& dq_out) {
  if (T <= 1e-9) {
    q_out = qf;
    dq_out.setZero();
    return;
  }
  const double u = std::clamp(t / T, 0.0, 1.0);
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  const double u5 = u4 * u;
  const double s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
  const double ds = (30.0 * u2 - 60.0 * u3 + 30.0 * u4) / T;
  q_out = q0 + s * (qf - q0);
  dq_out = ds * (qf - q0);
}

controller_interface::return_type TeachReplayController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  updateJointStates();

  // Low-pass filter velocity estimate (matches joint_impedance_example_controller).
  constexpr double kAlpha = 0.99;
  dq_filtered_ = (1.0 - kAlpha) * dq_filtered_ + kAlpha * dq_;

  // Pull pending trajectory only while we are not actively replaying. Swapping
  // mid-replay would discard progress and confuse the orchestrator.
  if (has_new_trajectory_.load() &&
      static_cast<Phase>(phase_.load()) == Phase::TEACH) {
    std::unique_lock<std::mutex> lk(trajectory_mutex_, std::try_to_lock);
    if (lk.owns_lock() && next_trajectory_) {
      active_trajectory_ = next_trajectory_;
      next_trajectory_.reset();
      has_new_trajectory_.store(false);
      replay_elapsed_ = 0.0;
    }
  }

  Vector7d q_d = q_;
  Vector7d dq_d = Vector7d::Zero();

  const auto phase = static_cast<Phase>(phase_.load());

  // ------------------------------------------------------------------
  // PRE_ROLL: min-jerk move from the pose at REPLAY entry to traj[0].
  // ------------------------------------------------------------------
  if (phase == Phase::PRE_ROLL && active_trajectory_) {
    const auto& traj = *active_trajectory_;
    if (pre_roll_duration_ <= 0.0) {
      // First tick of pre-roll: snapshot start pose and size the duration
      // from the per-joint distance so the move respects max_velocity.
      pre_roll_q_start_ = q_;
      pre_roll_elapsed_ = 0.0;
      const Vector7d delta = (traj.positions.front() - pre_roll_q_start_).cwiseAbs();
      const double max_delta = delta.maxCoeff();
      // For a quintic min-jerk profile q(u) = q0 + (10u^3-15u^4+6u^5)(qf-q0)
      // the peak |dq/dt| is 15/8 * |qf - q0| / T = 1.875 * dq / T. To bound
      // the peak per-joint velocity by move_to_start_max_velocity_, set
      //     T >= 1.875 * max_delta / max_velocity
      constexpr double kMinJerkPeakFactor = 1.875;
      pre_roll_duration_ = std::max(
          move_to_start_min_duration_,
          kMinJerkPeakFactor * max_delta / move_to_start_max_velocity_);
      RCLCPP_INFO(get_node()->get_logger(),
                  "Pre-roll: max joint delta %.3f rad, duration %.2f s "
                  "(peak velocity bound %.2f rad/s)",
                  max_delta, pre_roll_duration_, move_to_start_max_velocity_);
    }

    pre_roll_elapsed_ += period.seconds();
    minJerk(pre_roll_q_start_, traj.positions.front(),
            pre_roll_elapsed_, pre_roll_duration_, q_d, dq_d);

    if (pre_roll_elapsed_ >= pre_roll_duration_) {
      // Pre-roll done: enter TRACKING. The replay_started signal only fires
      // here so external recorders begin capturing the actual replay, not
      // the move-to-start motion.
      phase_.store(static_cast<int>(Phase::TRACKING));
      replay_elapsed_ = 0.0;
      pre_roll_duration_ = 0.0;
      pre_roll_elapsed_ = 0.0;
      start_pending_ = true;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Pre-roll complete -> TRACKING (publishing replay_started)");
    }

    Vector7d tau =
        k_gains_.cwiseProduct(q_d - q_) + d_gains_.cwiseProduct(dq_d - dq_filtered_);
    for (int i = 0; i < kNumJoints; ++i) {
      command_interfaces_[i].set_value(tau(i));
    }
  }
  // ------------------------------------------------------------------
  // TRACKING: follow the dense recorded trajectory.
  // ------------------------------------------------------------------
  else if (phase == Phase::TRACKING && active_trajectory_) {
    replay_elapsed_ += period.seconds();
    const auto& traj = *active_trajectory_;
    const double t_end = traj.times.back();

    if (replay_elapsed_ >= t_end) {
      // Hold last point and finish.
      q_d = traj.positions.back();
      dq_d.setZero();
      hold_q_ = q_d;
      finish_pending_ = true;
      phase_.store(static_cast<int>(Phase::TEACH));
      RCLCPP_INFO(get_node()->get_logger(), "Replay finished; reverted to TEACH");
    } else {
      // Cubic Hermite interpolation between adjacent recorded samples.
      size_t i = 0;
      while (i + 1 < traj.times.size() && traj.times[i + 1] <= replay_elapsed_) {
        ++i;
      }
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
      const double dh00 = (6.0 * s2 - 6.0 * s) / h;
      const double dh10 = 3.0 * s2 - 4.0 * s + 1.0;
      const double dh01 = (-6.0 * s2 + 6.0 * s) / h;
      const double dh11 = 3.0 * s2 - 2.0 * s;

      q_d = h00 * traj.positions[i] + h10 * h * traj.velocities[i] +
            h01 * traj.positions[i + 1] + h11 * h * traj.velocities[i + 1];
      dq_d = dh00 * traj.positions[i] + dh10 * traj.velocities[i] +
             dh01 * traj.positions[i + 1] + dh11 * traj.velocities[i + 1];
    }

    Vector7d tau =
        k_gains_.cwiseProduct(q_d - q_) + d_gains_.cwiseProduct(dq_d - dq_filtered_);
    for (int i = 0; i < kNumJoints; ++i) {
      command_interfaces_[i].set_value(tau(i));
    }
  }
  // ------------------------------------------------------------------
  // TEACH (and any abort path): zero torque, internal gravity compensation.
  // ------------------------------------------------------------------
  else {
    for (auto& cmd : command_interfaces_) {
      cmd.set_value(0.0);
    }
    hold_q_ = q_;
    // Reset pre-roll state so the next REPLAY recomputes duration.
    pre_roll_duration_ = 0.0;
    pre_roll_elapsed_ = 0.0;
    replay_elapsed_ = 0.0;
  }

  // Edge-triggered notifications outside the phase branches.
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
