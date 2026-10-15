"""Manual teleoperation interface for collecting demonstrations.

Use keyboard to control the robot and record successful demonstrations.
This is the "minimal demonstration collection system" requested.
"""

import sys
sys.path.insert(0, 'backend')

import numpy as np
from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv
import pickle
from pathlib import Path
from datetime import datetime


class TeleopCollector:
    """Collect demonstrations via keyboard teleoperation."""

    def __init__(self, env):
        self.env = env
        self.recording = False
        self.observations = []
        self.actions = []
        self.rewards = []

        # Control mode
        self.control_scale = 0.05
        self.left_gripper = 0.0
        self.right_gripper = 0.0

    def get_keyboard_action(self):
        """Get action from keyboard input.

        Controls:
            w/s: left arm joint 0 (base rotation)
            a/d: left arm joint 1 (shoulder)
            q/e: left arm joint 2 (elbow)
            z/x: left gripper open/close

            i/k: right arm joint 0
            j/l: right arm joint 1
            u/o: right arm joint 2
            n/m: right gripper open/close

            r: start/stop recording
            space: reset episode
            esc: quit
        """
        print("\nCurrent state:")
        print(f"  Left gripper: {'closed' if self.left_gripper > 0.5 else 'open'}")
        print(f"  Right gripper: {'closed' if self.right_gripper > 0.5 else 'open'}")
        print(f"  Recording: {self.recording}")
        print("\nControls:")
        print("  Left arm:  w/s (base), a/d (shoulder), q/e (elbow), z/x (gripper)")
        print("  Right arm: i/k (base), j/l (shoulder), u/o (elbow), n/m (gripper)")
        print("  r: record, space: reset, esc: quit")

        cmd = input("\nEnter command: ").strip().lower()

        # Initialize action
        action = np.zeros(16)

        # Left arm controls
        if 'w' in cmd:
            action[0] = self.control_scale
        elif 's' in cmd:
            action[0] = -self.control_scale
        if 'a' in cmd:
            action[1] = self.control_scale
        elif 'd' in cmd:
            action[1] = -self.control_scale
        if 'q' in cmd:
            action[2] = self.control_scale
        elif 'e' in cmd:
            action[2] = -self.control_scale

        # Right arm controls
        if 'i' in cmd:
            action[7] = self.control_scale
        elif 'k' in cmd:
            action[7] = -self.control_scale
        if 'j' in cmd:
            action[8] = self.control_scale
        elif 'l' in cmd:
            action[8] = -self.control_scale
        if 'u' in cmd:
            action[9] = self.control_scale
        elif 'o' in cmd:
            action[9] = -self.control_scale

        # Grippers
        if 'z' in cmd:
            self.left_gripper = 0.0
        elif 'x' in cmd:
            self.left_gripper = 1.0
        if 'n' in cmd:
            self.right_gripper = 0.0
        elif 'm' in cmd:
            self.right_gripper = 1.0

        action[14] = self.left_gripper
        action[15] = self.right_gripper

        # Meta commands
        meta = None
        if 'r' in cmd:
            meta = 'record'
        elif ' ' in cmd:
            meta = 'reset'
        elif 'esc' in cmd:
            meta = 'quit'

        return action, meta

    def collect(self, save_dir="demonstrations"):
        """Run teleoperation collection loop."""
        print("=== Teleoperation Demonstration Collector ===")
        print("Starting episode...")

        obs, info = self.env.reset()
        trajectories = []
        done = False

        while True:
            # Display current state
            print(f"\nStep {len(self.observations)}")
            print(f"  Left TCP: {obs[14:17]}")
            print(f"  Right TCP: {obs[17:20]}")
            print(f"  Left obj: {obs[20:23]}")
            print(f"  Right obj: {obs[23:26]}")

            # Get keyboard action
            action, meta = self.get_keyboard_action()

            # Handle meta commands
            if meta == 'quit':
                print("\nQuitting...")
                break
            elif meta == 'reset':
                print("\nResetting episode...")
                obs, info = self.env.reset()
                self.recording = False
                self.observations = []
                self.actions = []
                self.rewards = []
                continue
            elif meta == 'record':
                self.recording = not self.recording
                if self.recording:
                    print("\n[REC] Recording started!")
                else:
                    print("\n[STOP] Recording stopped!")
                    if len(self.observations) > 0:
                        # Save trajectory
                        trajectory = {
                            'observations': np.array(self.observations),
                            'actions': np.array(self.actions),
                            'rewards': np.array(self.rewards),
                            'episode_length': len(self.observations),
                            'success': True  # User confirms by stopping recording
                        }
                        trajectories.append(trajectory)
                        print(f"Trajectory saved ({len(self.observations)} steps)")

                        # Save to file
                        save_path = Path(save_dir)
                        save_path.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        demo_file = save_path / f"teleop_demo_{timestamp}.pkl"
                        with open(demo_file, 'wb') as f:
                            pickle.dump(trajectories, f)
                        print(f"Saved to: {demo_file}")

                    self.observations = []
                    self.actions = []
                    self.rewards = []
                continue

            # Execute action
            if self.recording:
                self.observations.append(obs)
                self.actions.append(action)

            obs, reward, terminated, truncated, info = self.env.step(action)

            if self.recording:
                self.rewards.append(reward)

            if terminated or truncated:
                print(f"\nEpisode ended (success: {info.get('episode_success', False)})")
                obs, info = self.env.reset()

        self.env.close()
        print(f"\nCollected {len(trajectories)} trajectories total")


if __name__ == '__main__':
    env = DualArmPickPlaceEnv(render_mode='human')
    collector = TeleopCollector(env)
    collector.collect()
