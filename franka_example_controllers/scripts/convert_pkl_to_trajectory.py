#!/usr/bin/env python3
# Copyright (c) 2024 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to convert pickle/joblib trajectory file to trajectory format for trajectory following controller.

Usage:
    python3 convert_pkl_to_trajectory.py <input.pkl> <output.trajectory>

The script expects the file to contain a dictionary with 'arm_joint_positions' key.
The 'arm_joint_positions' can be:
    - A numpy array (2D array where each row is a frame with 7 joint positions)
    - A list of lists/arrays (each element is a frame with 7 joint positions)
    - A dict (keys are frame indices, values are arrays/lists with 7 joint positions)

The script supports both pickle and joblib formats.
"""

import pickle
import sys
import argparse
import numpy as np


def convert_pkl_to_trajectory(input_file, output_file):
    """Convert pickle file to trajectory YAML format."""
    try:
        # Try using pickle first
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        # If pickle fails, try joblib
        try:
            import joblib
            data = joblib.load(input_file)
        except ImportError:
            print(f"Error loading pickle file: {e}")
            print("Note: If using joblib format, install joblib: pip install joblib")
            return False
        except Exception as e2:
            print(f"Error loading file with both pickle and joblib: {e}, {e2}")
            return False
    
    if 'arm_joint_positions' not in data:
        print("Error: 'arm_joint_positions' key not found in pickle file")
        return False
    
    arm_joint_positions = data['arm_joint_positions']
    
    # Handle numpy arrays
    if isinstance(arm_joint_positions, np.ndarray):
        # Convert numpy array to list of lists
        trajectory = arm_joint_positions.tolist()
    elif isinstance(arm_joint_positions, dict):
        # If it's a dict, extract values in order
        frames = sorted(arm_joint_positions.keys())
        trajectory = [arm_joint_positions[frame] for frame in frames]
        # Convert numpy arrays in dict values if needed
        trajectory = [np.array(waypoint).tolist() if isinstance(waypoint, np.ndarray) else waypoint 
                      for waypoint in trajectory]
    elif isinstance(arm_joint_positions, list):
        trajectory = arm_joint_positions
        # Convert numpy arrays in list if needed
        trajectory = [np.array(waypoint).tolist() if isinstance(waypoint, np.ndarray) else waypoint 
                      for waypoint in trajectory]
    else:
        print(f"Error: 'arm_joint_positions' has unsupported type: {type(arm_joint_positions)}")
        print("Expected: numpy array, list, or dict")
        return False
    
    # Write trajectory to output file (simple text format: one waypoint per line)
    try:
        with open(output_file, 'w') as f:
            f.write("# Trajectory file for trajectory_following_joint_impedance_controller\n")
            f.write("# Format: 7 joint positions per line (space-separated)\n")
            f.write(f"# Total waypoints: {len(trajectory)}\n")
            
            for idx, waypoint in enumerate(trajectory):
                # Convert to list if it's a numpy array
                if isinstance(waypoint, np.ndarray):
                    waypoint = waypoint.tolist()
                
                # Ensure it's a list/array-like with 7 elements
                try:
                    waypoint_list = list(waypoint)
                except (TypeError, ValueError):
                    print(f"Warning: Waypoint {idx} cannot be converted to list. Skipping.")
                    continue
                
                if len(waypoint_list) != 7:
                    print(f"Warning: Waypoint {idx} has {len(waypoint_list)} values, expected 7. Skipping.")
                    continue
                
                # Write 7 joint positions
                f.write(f"{waypoint_list[0]:.6f} {waypoint_list[1]:.6f} {waypoint_list[2]:.6f} "
                       f"{waypoint_list[3]:.6f} {waypoint_list[4]:.6f} {waypoint_list[5]:.6f} "
                       f"{waypoint_list[6]:.6f}\n")
        
        print(f"Successfully converted {len(trajectory)} waypoints to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert pickle trajectory file to YAML format for trajectory following controller'
    )
    parser.add_argument('input_file', help='Input pickle file path')
    parser.add_argument('output_file', help='Output trajectory file path')
    
    args = parser.parse_args()
    
    success = convert_pkl_to_trajectory(args.input_file, args.output_file)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

