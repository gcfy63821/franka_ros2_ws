# Teach & Replay 使用文档

`TeachReplayController` + `teach_replay_orchestrator.py` 的完整使用流程：手动拖动机械臂示教轨迹，再让机械臂自动重放并采集视频与关节数据。

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

```bash
ros2 launch realsense2_camera rs_launch.py \
  config_file:=/home/robot/franka_ros2_ws/install/franka_bringup/share/franka_bringup/config/realsense_d455.yaml
```

默认配置使用 D455 彩色流 `640x480x30`，图像 topic 为 `/camera/camera/color/image_raw`。如果用其它相机，启动 orchestrator 时加 `--image_topic <topic>`。

### 终端 3：键盘编排节点

```bash
ros2 run franka_example_controllers teach_replay_orchestrator.py
# 可选参数:
#   --fps 30                 输出视频帧率
#   --record_rate 100        teach/replay 时关节角采样率（Hz）
#   --trajectory_smoothing_window 11
#                            replay 前对示教关节轨迹做移动平均平滑；<=1 关闭
#   --output_dir ~/robot_recordings
#   --image_topic /camera/camera/color/image_raw
#   --no_video               关掉视频录制（仅记关节）
```

> 此终端会把键盘切成 raw mode，所以**所有按键都要在该终端窗口里按**。

---

## 4. 键位

| 键 | 阶段 | 动作 |
|---|---|---|
| `t` | IDLE / READY | 进入 TEACH，零力矩；新建 session 文件夹 |
| `o` | TEACHING | 张开夹爪（调用 `Move` action，width=0.08m）+ 记录事件 |
| `c` | TEACHING | 闭合夹爪（调用 `Grasp` action，force=20N）+ 记录事件 |
| `s` | TEACHING | 停止示教，保存 teach 数据 → READY |
| `r` | READY | 发布轨迹 + 切到 REPLAY 模式，自动开始录像和记录关节 |
| `h` | any | 夹爪 homing（标定行程） |
| `q` | any | 退出 |

REPLAY 自动结束后状态回到 READY，可再按 `r` 重放、或按 `t` 重新示教。

---

## 5. 一次完整 session 的典型操作

1. 终端 1、2、3 全部启动，确认 orchestrator 打印出 `Gripper action clients created`
2. 在终端 3 按 `t`：日志显示 `State: IDLE -> TEACHING`，机械臂变软
3. 用手把机械臂拖到起始姿态，需要时按 `o` / `c` 切换夹爪
4. 拖完整条轨迹后按 `s`：日志输出 `Teach saved: N samples, duration X.Xs`
5. 把工件 / 相机视野复位
6. 按 `r`：机械臂开始重放，终端 1 控制器日志会显示 `Mode -> REPLAY`，轨迹结束时显示 `Replay finished; reverted to TEACH`
7. 终端 3 自动保存 `replay/` 下的视频和关节数据

---

## 6. 输出文件

```
~/robot_recordings/session_YYYYMMDD_HHMMSS/
├── teach/
│   ├── joint_trajectory.npz      # timestamps, joint_positions[N,7], gripper_widths[N,2]
│   └── gripper_events.npz        # relative_times[K], actions[K]  ("open"/"close")
└── replay/
    ├── recording.avi             # 视频
    ├── joint_trajectory.npz      # 实际执行的关节角
    └── gripper_events.npz        # 重放时实际下达夹爪命令的时间戳
```

加载示例：
```python
import numpy as np
d = np.load("~/robot_recordings/session_.../teach/joint_trajectory.npz")
print(d["joint_positions"].shape)   # (N, 7)
print(d["timestamps"][-1])          # 总时长
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

`franka_bringup/config/controllers.yaml` 里 `teach_replay_controller` 的 `k_gains` / `d_gains`：

```yaml
k_gains: [200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 50.0]
d_gains: [ 20.0,  20.0,  20.0,  20.0,  10.0,  10.0,  5.0]
```

- 重放跟踪不够紧 → 调高 `k_gains`（每个关节按 1.5–2× 递增试）
- 重放有抖动或过冲 → 调高 `d_gains`，或降低 `k_gains`
- TEACH 阶段感觉不够"软" → 与控制器无关，TEACH 始终发零力矩；阻力来自机器人内部摩擦

修改后必须 `colcon build --packages-select franka_bringup` 并重启终端 1。

---

## 9. 已知约束

- **REPLAY 起始位姿**：控制器在 REPLAY 时直接把第一点当参考，若当前位姿离 `traj[0]` 很远会有冲击。建议每次按 `r` 前手动把机械臂拖回示教起点附近，或后续在控制器里加一段从当前位姿到 `traj[0]` 的 min-jerk 过渡。
- **夹爪 namespace**：当前假设 `franka.config.yaml` 中 `namespace: ""`。若改了 namespace，需要修改 orchestrator 中 `/franka_gripper/...` 路径。
- **录像时机**：视频写入由 image 回调驱动（实际帧率 ≈ 相机帧率），但 `VideoWriter` 的 fps 头由 `--fps` 决定；如果两者差距大，回放视频会快/慢于现实。把 `--fps` 设成相机实际帧率即可。
- **数据时间基准**：teach/replay 双方的 `relative_times` 都以 `time.time()` 为基准（wall clock），不是 ROS time。
