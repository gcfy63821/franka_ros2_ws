# EE Pose Control Replay 使用文档

使用 `EePoseReplayController` + `ee_control_replay.py`，读取每个
`traj_*/replay/EE_pose_FK.npz` 中的末端位姿，用 Franka Cartesian pose command
interface 重放，并把新的状态记录到 `traj_*/FK_state/`。

每条轨迹开始前，控制器会先用 min-jerk 轨迹从当前 TCP 位姿平稳过渡到
FK 文件第一帧，然后才发布 `replay_started` 并开始记录。

## 1. 构建

```bash
cd ~/franka_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select franka_example_controllers franka_bringup \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## 2. 启动控制器

终端 1：

```bash
source ~/franka_ros2_ws/install/setup.bash
ros2 launch franka_bringup ee_control_replay.launch.py
```

该 launch 会启动 `ee_pose_replay_controller`，使用以下 topic：

| 名称 | 类型 | 用途 |
|---|---|---|
| `/ee_pose_replay/mode` | `std_msgs/String` | `"hold"` / `"replay"` |
| `/ee_pose_replay/trajectory` | `trajectory_msgs/MultiDOFJointTrajectory` | FK 末端位姿轨迹 |
| `/ee_pose_replay/replay_started` | `std_msgs/Bool` | pre-roll 结束、正式 replay 开始 |
| `/ee_pose_replay/replay_finished` | `std_msgs/Bool` | replay 完成 |

## 3. 批量 replay

终端 2：

```bash
source ~/franka_ros2_ws/install/setup.bash
ros2 run franka_example_controllers ee_control_replay.py \
  --data_dir ~/robot_recordings/at
```

非 test 模式会自动按 `traj_0, traj_1, ...` 顺序重放所有包含
`replay/EE_pose_FK.npz` 的轨迹。

## 4. Test 模式

```bash
ros2 run franka_example_controllers ee_control_replay.py \
  --data_dir ~/robot_recordings/at \
  --test
```

`--test` 下每次只 replay 一条轨迹；每条开始前终端会提示按 Enter，输入
`q` 退出。

## 5. 输入文件

每条轨迹至少需要：

```text
traj_N/
└── replay/
    └── EE_pose_FK.npz
```

`EE_pose_FK.npz` 支持：
- `timestamps` 或 `relative_times`
- 位姿数组 key 自动识别：`end_effector_poses`、`ee_pose`、`ee_poses`、
  `tcp_pose`、`tcp_poses`、`poses`，也可用 `--pose_key` 指定
- 位姿格式：`(N, 7)`，顺序为 `x, y, z, qx, qy, qz, qw`

夹爪事件默认自动读取：
1. `traj_N/replay/gripper_events.npz`
2. 若不存在，则读取 `traj_N/teach/gripper_events.npz`

可用 `--gripper_source replay|teach|none` 指定来源或关闭夹爪事件。

## 6. 输出文件

每条 replay 完成后写入：

```text
traj_N/
└── FK_state/
    ├── tcp_pose.npz        # timestamps, tcp_pose[N,7]
    ├── joint_velocity.npz  # timestamps, joint_velocity[N,7]
    ├── joint_pos.npz       # timestamps, joint_pos[N,7]
    └── gripper_events.npz  # 实际下发的夹爪事件
```

其中 `tcp_pose` 顺序为 `x, y, z, qx, qy, qz, qw`。

## 7. 常用参数

```bash
ros2 run franka_example_controllers ee_control_replay.py \
  --data_dir ~/robot_recordings/at \
  --record_rate 100 \
  --trajectory_smoothing_window 5 \
  --input_rate 100
```

- `--record_rate`：记录 `FK_state` 的采样频率
- `--trajectory_smoothing_window`：对 FK 位姿中的 position 做移动平均平滑；
  `<=1` 关闭
- `--input_rate`：当 `EE_pose_FK.npz` 没有 timestamp 时使用的默认频率
- `--finish_timeout`：等待单条轨迹结束的超时时间；`<=0` 表示一直等待

## 8. 控制器安全参数

`franka_bringup/config/controllers.yaml`：

```yaml
ee_pose_replay_controller:
  ros__parameters:
    move_to_start: true
    move_to_start_min_duration: 4.0
    move_to_start_max_translation_velocity: 0.05  # m/s
    move_to_start_max_rotation_velocity: 0.5      # rad/s
```

pre-roll 太快时，降低 `move_to_start_max_translation_velocity` /
`move_to_start_max_rotation_velocity`，或增大 `move_to_start_min_duration`。
