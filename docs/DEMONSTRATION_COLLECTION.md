# Demonstration Collection System for Dual-Arm Pick-Place

## Overview

This document describes the minimal demonstration collection system set up for the dual-arm pick-and-place task (选项B).

## Problem Context

The dual-arm pick-place task requires expert demonstrations for imitation learning (BC + RL). Initial attempts to create automated expert policies encountered challenges:

1. **IK-based expert** (`backend/dual_arm_ik_expert.py`): Simulation state divergence between expert's internal `self.data` and environment's `env.data` caused incorrect action commands
2. **Scripted policy** (`backend/dual_arm_scripted_policy.py`): Hard-coded joint configurations didn't match actual robot workspace (TCP starts at 1.8m height, objects at 0.15m)
3. **Random exploration** (`tools/collect_random_demos.py`): 0% success rate after 50 attempts - task is too complex for random actions

## Solution: Teleoperation Interface

Created a keyboard-based teleoperation system that allows manual control and demonstration recording:

### File: `tools/teleop_collect.py`

**Features:**
- Real-time keyboard control of both robot arms
- Manual gripper open/close control
- Start/stop recording with 'r' key
- Episode reset with spacebar
- Automatic trajectory saving to pickle format

**Controls:**
```
Left arm:  w/s (joint 0), a/d (joint 1), q/e (joint 2), z/x (gripper)
Right arm: i/k (joint 0), j/l (joint 1), u/o (joint 2), n/m (gripper)
Meta:      r (record toggle), space (reset), esc (quit)
```

**Usage:**
```bash
python tools/teleop_collect.py
```

**Output:**
- Saves demonstrations to `demonstrations/teleop_demo_YYYYMMDD_HHMMSS.pkl`
- Format: list of dicts with keys: `observations`, `actions`, `rewards`, `episode_length`, `success`

## Alternative Collection Tools

### 1. Original demonstration collector (`tools/collect_demonstrations.py`)
- Supports scripted policy for pick-place task
- Currently has 0% success rate due to workspace mismatch
- Can be improved by:
  - Tuning joint configurations in `backend/dual_arm_scripted_policy.py`
  - Using IK with shared environment state
  - Recording good configurations from teleoperation

### 2. Random exploration (`tools/collect_random_demos.py`)
- Filters successful episodes from random actions
- Not practical for this task (0% success rate observed)
- Could work for simpler tasks or with better exploration strategy

## Data Format

All collectors output the same format:

```python
trajectory = {
    'observations': np.ndarray,  # Shape: (T, 28) - T timesteps
    'actions': np.ndarray,        # Shape: (T, 16)
    'rewards': np.ndarray,        # Shape: (T,)
    'episode_length': int,        # Number of timesteps
    'success': bool              # Whether episode succeeded
}
```

**Observation space (28D):**
- [0:14]: Joint positions (7 per arm)
- [14:17]: Left TCP position (x, y, z)
- [17:20]: Right TCP position (x, y, z)
- [20:23]: Left object position (x, y, z)
- [23:26]: Right object position (x, y, z)
- [26:28]: Grasp states (binary per arm)

**Action space (16D):**
- [0:14]: Joint position deltas (7 per arm)
- [14:16]: Gripper commands (0=open, 1=closed)

## Next Steps

1. **Collect demonstrations**:
   ```bash
   python tools/teleop_collect.py
   ```
   Target: 50-100 successful demonstrations

2. **Inspect collected data**:
   ```bash
   python tools/collect_demonstrations.py --inspect demonstrations/teleop_demo_*.pkl
   ```

3. **Train BC model**:
   - Implement BC trainer in `backend/bc_trainer.py`
   - Train on collected demonstrations
   - Evaluate policy success rate

4. **Combine with RL**:
   - Use BC policy as initialization
   - Fine-tune with SAC/TD3
   - Compare against pure RL baseline

## Known Issues

1. **Teleoperation challenges**:
   - Keyboard control is imprecise (discrete steps)
   - No visual feedback in terminal-only mode
   - Requires practice to achieve successful episodes

2. **Potential improvements**:
   - Add joystick/gamepad support
   - Implement replay buffer visualization
   - Add trajectory smoothing post-processing
   - Create GUI-based kinesthetic teaching interface

3. **IK expert could be fixed** by:
   - Passing `env.data` directly to expert instead of maintaining separate simulation
   - Or using IK only for target computation, not for maintaining state

## Conclusion

The minimal demonstration collection system is **teleoperation-based** (`tools/teleop_collect.py`). While not as automated as a scripted expert, it provides a working path forward for collecting the demonstrations needed for BC + RL training.

For future work, consider:
- Improving the scripted policy with correct joint configurations
- Fixing the IK expert's simulation state management
- Adding more sophisticated teleoperation interfaces (GUI, gamepad)
