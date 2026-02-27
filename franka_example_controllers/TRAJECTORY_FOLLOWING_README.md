# Trajectory Following Joint Impedance Controller

This controller loads a trajectory from a file and follows it using joint impedance control with minimum jerk planning.

## Features

- Loads trajectory from a text file (converted from pkl format)
- Moves toward the first waypoint using minimum jerk trajectory planning
- Follows the trajectory at a configurable rate (default: 30 Hz)
- Uses joint impedance control for smooth, compliant motion

## Usage

### 1. Convert PKL file to trajectory format

First, convert your pickle file to the trajectory format:

```bash
python3 $(find franka_example_controllers)/scripts/convert_pkl_to_trajectory.py <input.pkl> <output.trajectory>
```
/home/robot/workspace/franka_control/src/trajectory_ep0008.pkl

The script expects the pickle file to contain a dictionary with `arm_joint_positions` key, where each frame contains 7 joint positions.

### 2. Set trajectory file path

You can set the trajectory file path in two ways:

**Option A: Edit the config file**
Edit `franka_gazebo_bringup/config/franka_gazebo_controllers.yaml` and set the `trajectory_file` parameter:

```yaml
trajectory_following_joint_impedance_controller:
  ros__parameters:
    trajectory_file: "/path/to/your/trajectory.trajectory"
```

**Option B: Set via ROS parameter (after launch)**
```bash
ros2 param set /controller_manager/trajectory_following_joint_impedance_controller trajectory_file "/path/to/your/trajectory.trajectory"
```

### 3. Launch in Gazebo

```bash
ros2 launch franka_gazebo_bringup gazebo_trajectory_following_joint_impedance_controller_example.launch.py
```

Or with a trajectory file argument (if you modify the launch file to accept it):

```bash
ros2 launch franka_gazebo_bringup gazebo_trajectory_following_joint_impedance_controller_example.launch.py trajectory_file:=/path/to/trajectory.trajectory
```
### 4 Launch in real robot:

```bash
ros2 launch franka_bringup trajectory_following_joint_impedance_controller.launch.py
```

## Configuration Parameters

- `trajectory_file`: Path to the trajectory file (required)
- `trajectory_rate`: Trajectory following rate in Hz (default: 30.0)
- `approach_duration`: Time in seconds to reach the first waypoint using minimum jerk (default: 8.0)
- `k_gains`: Stiffness gains for each joint (7 values)
- `d_gains`: Damping gains for each joint (7 values)
- `gazebo`: Set to `true` for Gazebo simulation (default: false)

## Trajectory File Format

The trajectory file is a simple text format with one waypoint per line. Each line contains 7 space-separated joint positions (in radians):

```
0.0 -0.785 0.0 -2.356 0.0 1.571 0.785
0.1 -0.785 0.0 -2.356 0.0 1.571 0.785
0.2 -0.785 0.0 -2.356 0.0 1.571 0.785
...
```

Lines starting with `#` are treated as comments and ignored.

## Controller Behavior

1. **Initialization**: On activation, the controller records the current joint positions as the starting point.

2. **Approach Phase**: The controller uses minimum jerk trajectory planning to smoothly move from the current position to the first waypoint. This phase takes `approach_duration` seconds.

3. **Trajectory Following**: Once the first waypoint is reached (within 0.05 rad threshold), the controller starts following the trajectory at the specified rate (default 30 Hz).

4. **End of Trajectory**: When the trajectory ends, the controller holds the last waypoint position.

## Minimum Jerk Planning

The minimum jerk trajectory uses a 5th-order polynomial that minimizes jerk (third derivative of position). The trajectory is computed as:

```
s = t / duration  (normalized time [0, 1])
poly = 10*s³ - 15*s⁴ + 6*s⁵
q(t) = q_start + poly * (q_end - q_start)
```

This ensures smooth, natural motion with zero velocity and acceleration at the start and end points.

