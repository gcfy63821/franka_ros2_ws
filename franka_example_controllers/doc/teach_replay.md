# Teach & Replay 使用文档

`TeachReplayController` + `teach_replay_orchestrator.py` 的完整使用流程：手动拖动机械臂示教轨迹，再让机械臂自动重放并采集视频、关节位置、关节速度和末端位姿数据。

---

## 1. 网络准备（如需）

机器人主机若要走本地代理上网，先在你本机起反向 SSH 隧道（把本机 7890 端口转发到机器人主机 7897）：

```bash
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -NR 7890:127.0.0.1:7897 robot
```

> `-N` 不开 shell；`-R` 反向端口转发；`ServerAliveInterval/CountMax` 防止链路因空闲被断。

---

## 2. 构建工作区

```bash
cd ~/crq_ws/dexx/franka_ros2_control
source /opt/ros/humble/setup.bash      # 或对应的 ROS 2 发行版
colcon build --packages-select franka_example_controllers franka_bringup \
             --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

每次修改控制器源码或 YAML 后都需要重新 build + source。

---

## 3. 启动流程

需要三个终端（每个终端都先 `source install/setup.bash`）。

### 终端 1：机器人 + 控制器

```bash
ros2 launch franka_bringup teach_replay.launch.py
```

`franka.launch.py` 会被自动 include，附带启动：
- `ros2_control_node`（1 kHz 实时控制循环）
- `franka_robot_state_broadcaster` + `joint_state_broadcaster`
- `franka_gripper`（暴露 `/franka_gripper/move`、`/grasp`、`/homing` 三个 action）
- `teach_replay_controller`（默认在 TEACH 模式 = 零力矩 = 可手动拖动）

### 终端 2：相机（如需录像）

务必用本仓库安装后的 D455 配置启动相机：

```bash
ros2 launch realsense2_camera rs_launch.py \
  config_file:=/home/robot/franka_ros2_ws/install/franka_bringup/share/franka_bringup/config/realsense_d455.yaml
```

默认配置使用 D455 彩色流 `640x480x30`，关闭 depth / infra / IMU，并发布 raw + compressed 彩色图像。orchestrator 默认订阅 `/camera/camera/color/image_raw/compressed`，避开 Python 订阅 raw 大消息时的掉帧问题。如果用其它相机，启动 orchestrator 时加 `--image_topic <topic>`；如果必须订阅 raw 图像，再加 `--raw_image`。

启动后建议先确认实际参数和帧率：

```bash
ros2 param get /camera/camera rgb_camera.color_profile
ros2 param get /camera/camera enable_depth
ros2 param get /camera/camera camera.color.image_raw.enable_pub_plugins
ros2 topic hz /camera/camera/color/image_raw/compressed
```

期望结果是 `rgb_camera.color_profile = 640x480x30`、`enable_depth = False`、插件包含 `image_transport/compressed`、compressed topic 接近 30 Hz。若直接用 RealSense 默认启动命令，可能会跑成 `1280x720x30` 且 depth 开启；若订阅 raw `/camera/camera/color/image_raw`，Python 端也可能因为大消息传输/反序列化掉到十几 Hz，最终 replay 视频只能记录到十几 Hz。

### 终端 3：键盘编排节点

```bash
ros2 run franka_example_controllers teach_replay_orchestrator.py
enter save data folder name: task1
# 可选参数:
#   --fps 30                 输出视频帧率
#   --record_rate 100        teach/replay 时关节角采样率（Hz）
#   --trajectory_smoothing_window 11
#                            replay 前对示教关节轨迹做移动平均平滑；<=1 关闭
#   --output_dir ~/robot_recordings
#   --image_topic /camera/camera/color/image_raw/compressed
#   --raw_image              订阅 raw sensor_msgs/Image；默认订阅 compressed 图像
#   --no_video               关掉视频录制（仅记关节）
```

> 此终端会把键盘切成 raw mode，所以**所有按键都要在该终端窗口里按**。
> 启动时先输入本次采集保存的任务文件夹名，例如 `task1`；脚本会在 `~/robot_recordings/task1/` 中继续创建下一个 `traj_N`。

---

## 4. 键位

| 键 | 阶段 | 动作 |
|---|---|---|
| `t` | IDLE / READY | 进入 TEACH，零力矩；新建下一个 `traj_N` 文件夹 |
| `o` | TEACHING | 张开夹爪（调用 `Move` action，width=0.08m）+ 记录事件 |
| `c` | TEACHING | 闭合夹爪（调用 `Grasp` action，force=20N）+ 记录事件 |
| `s` | TEACHING | 停止示教，保存 teach 数据 → READY |
| `r` | READY | 发布轨迹 + 切到 REPLAY 模式，自动开始录像和记录 replay 数据 |
| `1` / `s` | replay 结束后 | 保存本次 replay 数据 |
| `2` / `d` | replay 结束后 | 删除这条 replay 和对应 teach 轨迹 |
| `3` / `r` | replay 结束后 | 丢弃本次 replay；回到 READY 后再按一次 `r` 重新 replay |
| `h` | any | 夹爪 homing（标定行程） |
| `q` | any | 退出 |

REPLAY 自动结束后会停在 review 状态，必须选择 save / dump / replay again 之一。

---

## 5. 一次完整 session 的典型操作

1. 终端 1、2、3 全部启动，在终端 3 输入保存文件夹名，例如 `task1`，确认 orchestrator 打印出 `Gripper action clients created`
2. 在终端 3 按 `t`：日志显示 `State: IDLE -> TEACHING`，机械臂变软
3. 用手把机械臂拖到起始姿态，需要时按 `o` / `c` 切换夹爪
4. 拖完整条轨迹后按 `s`：日志输出 `Teach saved: N samples, duration X.Xs`
5. 把工件 / 相机视野复位
6. 按 `r`：机械臂开始重放，终端 1 控制器日志会显示 `Mode -> REPLAY`，轨迹结束时显示 `Replay finished; reverted to TEACH`
7. replay 结束后终端 3 会提示选择：按 `1`/`s` 保存，按 `2`/`d` 删除整条轨迹，按 `3`/`r` 丢弃本次 replay 并准备重录

---

## 6. 输出文件

```
~/robot_recordings/task1/
├── traj_0/
│   ├── teach/
│   │   ├── joint_trajectory.npz  # timestamps, joint_positions[N,7], gripper_widths[N,2]
│   │   └── gripper_events.npz    # relative_times[K], actions[K]  ("open"/"close")
│   └── replay/
│       ├── recording.avi         # 视频
│       ├── joint_trajectory.npz  # timestamps, joint_positions[N,7]
│       ├── joint_velocities.npz  # timestamps, joint_velocities[N,7]
│       ├── end_effector_pose.npz # timestamps, end_effector_poses[N,7] = x,y,z,qx,qy,qz,qw
│       └── gripper_events.npz    # 重放时实际下达夹爪命令的时间戳
└── traj_1/
    └── ...
```

加载示例：
```python
import numpy as np
d = np.load("~/robot_recordings/task1/traj_0/teach/joint_trajectory.npz")
print(d["joint_positions"].shape)   # (N, 7)
print(d["timestamps"][-1])          # 总时长

ee = np.load("~/robot_recordings/task1/traj_0/replay/end_effector_pose.npz")
print(ee["end_effector_poses"].shape)  # (N, 7), x,y,z,qx,qy,qz,qw
```

---

## 7. 关键 Topic / Action

| 名称 | 类型 | 用途 |
|---|---|---|
| `/teach_replay/mode` | `std_msgs/String` | `"teach"` / `"replay"` 切换控制器模式 |
| `/teach_replay/trajectory` | `trajectory_msgs/JointTrajectory` | 重放轨迹（含 `time_from_start`） |
| `/teach_replay/replay_started` | `std_msgs/Bool` | 控制器进入 REPLAY 时发一次 |
| `/teach_replay/replay_finished` | `std_msgs/Bool` | 重放完成时发一次 |
| `/joint_states` | `sensor_msgs/JointState` | 关节角来源（含 7 关节 + 2 finger） |
| `/franka_robot_state_broadcaster/robot_state` | `franka_msgs/FrankaRobotState` | replay 关节速度和末端位姿来源 |
| `/franka_gripper/move` | `franka_msgs/action/Move` | 张开夹爪到指定 width |
| `/franka_gripper/grasp` | `franka_msgs/action/Grasp` | 抓取（带力控） |
| `/franka_gripper/homing` | `franka_msgs/action/Homing` | 夹爪行程标定 |

如果不通过 orchestrator，也可以手动测试：
```bash
ros2 topic pub --once /teach_replay/mode std_msgs/msg/String "{data: teach}"
ros2 topic pub --once /teach_replay/mode std_msgs/msg/String "{data: replay}"
```

---

## 8. 调参

`franka_bringup/config/controllers.yaml` 里 `teach_replay_controller`：

```yaml
k_gains: [200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 50.0]
d_gains: [ 20.0,  20.0,  20.0,  20.0,  10.0,  10.0,  5.0]
move_to_start: true
move_to_start_min_duration: 4.0   # 秒，pre-roll 时长下限
move_to_start_max_velocity: 0.2   # rad/s，峰值每关节速度上限
```

- 重放跟踪不够紧 → 调高 `k_gains`（每个关节按 1.5–2× 递增试）
- 重放有抖动或过冲 → 调高 `d_gains`，或降低 `k_gains`
- TEACH 阶段感觉不够"软" → 与控制器无关，TEACH 始终发零力矩；阻力来自机器人内部摩擦
- pre-roll 太快/猛 → 调小 `move_to_start_max_velocity` 或调大 `move_to_start_min_duration`
- 想跳过 move-to-start（要求拖回起点后再按 `r`） → `move_to_start: false`

修改后必须 `colcon build --packages-select franka_bringup` 并重启终端 1。

---

## 9. REPLAY 内部状态机

按 `r` 后控制器在内部走两个阶段：

1. **PRE_ROLL**（move-to-start）：以当前位姿为起点，min-jerk 移动到 `traj[0]`。时长 = `max(min_duration, 1.875 × max_joint_dist / max_velocity)`（min-jerk 五次多项式的峰值速度系数是 15/8 = 1.875，所以 `max_velocity` 表示"峰值每关节角速度"）。控制器日志会打印实际时长。**这一阶段不会发 `/teach_replay/replay_started`**，所以 orchestrator 不会录像。
2. **TRACKING**：发布 `/teach_replay/replay_started`，开始 cubic Hermite 跟随密集轨迹。结束时发 `/teach_replay/replay_finished` 并自动回 TEACH。

任何阶段下，发 `mode: teach` 都会立即 abort 回零力矩。

## 10. 已知约束

- **夹爪 namespace**：当前假设 `franka.config.yaml` 中 `namespace: ""`。若改了 namespace，需要修改 orchestrator 中 `/franka_gripper/...` 路径。
- **录像时机**：视频写入由 image 回调驱动（实际帧率 ≈ `/camera/camera/color/image_raw/compressed` 的实际发布帧率），但 `VideoWriter` 的 fps 头由 `--fps` 决定；如果两者差距大，回放视频会快/慢于现实。若视频只有十几 Hz，先用 `ros2 topic hz /camera/camera/color/image_raw/compressed` 查相机源头帧率，再确认相机是用上面的 `realsense_d455.yaml` 启动的。
- **数据时间基准**：teach/replay 双方的 `timestamps` / `relative_times` 都以 `time.time()` 为基准（wall clock），不是 ROS time。
- **pre-roll 期间的夹爪状态**：示教记录里没有"起始夹爪状态"字段。如果示教第一个 gripper 事件不在 t=0，pre-roll 期间夹爪保持上一次操作的状态。如果需要严格匹配，按 `r` 前先手动把夹爪调到示教起始状态（或按 `h` homing）。
