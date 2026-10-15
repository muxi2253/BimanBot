#!/usr/bin/env python3
"""
Dual-arm pick-and-place Gym environment.

Task: Both arms simultaneously pick objects from the inner bin and place them
into the outer bin. This is a parallel sorting task where left and right arms
work independently but share the same source and target bins.
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


class DualArmPickPlaceEnv(gym.Env):
    """
    Dual-arm pick-and-place environment.

    Observation:
        - 14D joint positions (7 per arm)
        - 6D TCP positions (3D per arm)
        - 6D object positions (3D per object: left_object, right_object)
        - 2D grasp states (binary per arm: is_grasping)
        Total: 28D

    Action:
        - 14D joint position deltas (7 per arm)
        - 2D gripper commands (binary per arm: open/close)
        Total: 16D

    Reward:
        Dense reward based on:
        - Distance from TCP to object (pick phase)
        - Distance from object to target bin (place phase)
        - Success bonus when both objects placed

    Episode terminates when:
        - Both objects successfully placed in target bin
        - Max steps reached
        - Robot collision or instability
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        reward_type: str = "dense",
        success_threshold: float = 0.05,  # 5cm placement tolerance
    ):
        super().__init__()

        # Load scene
        if scene_path is None:
            repo = Path(__file__).parent.parent
            scene_path = str(repo / "scene" / "dual_arm_sorting_scene.xml")

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Rendering
        self.render_mode = render_mode
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        # Episode config
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.success_threshold = success_threshold
        self.current_step = 0

        # Get body/site IDs
        self.left_tcp_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp")
        self.right_tcp_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")

        self.source_bin_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "crate_source")
        self.target_bin_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "crate_target")

        # Objects to manipulate
        self.left_object_name = "left_object"
        self.right_object_name = "right_object"
        self.left_object_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.left_object_name)
        self.right_object_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.right_object_name)

        # Get freejoint qpos indices for objects
        self.left_obj_qpos = self._get_freejoint_qpos(self.left_object_id)
        self.right_obj_qpos = self._get_freejoint_qpos(self.right_object_id)

        # Target positions (center of target bin)
        self.target_pos = self.data.xpos[self.target_bin_id].copy()
        self.target_pos[2] += 0.15  # Raise 15cm above bin bottom

        # Observation and action spaces
        # Obs: 14 joints + 6 TCP + 6 obj + 2 grasp = 28D
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

        # Action: 14 joint deltas + 2 gripper = 16D
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32)

        # State tracking
        self.left_grasped = False
        self.right_grasped = False
        self.left_placed = False
        self.right_placed = False

    def _get_freejoint_qpos(self, body_id: int) -> int:
        """Get qpos index for body's freejoint."""
        jnt_adr = int(self.model.body_jntadr[body_id])
        if jnt_adr < 0:
            raise ValueError(f"Body {body_id} has no joint")
        return int(self.model.jnt_qposadr[jnt_adr])

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        # Joint positions (14D)
        qpos = self.data.qpos[:14].copy()

        # TCP positions (6D)
        left_tcp = self.data.site_xpos[self.left_tcp_id].copy()
        right_tcp = self.data.site_xpos[self.right_tcp_id].copy()

        # Object positions (6D)
        left_obj = self.data.xpos[self.left_object_id].copy()
        right_obj = self.data.xpos[self.right_object_id].copy()

        # Grasp states (2D)
        grasp = np.array([float(self.left_grasped), float(self.right_grasped)])

        obs = np.concatenate([qpos, left_tcp, right_tcp, left_obj, right_obj, grasp])
        return obs.astype(np.float32)

    def _compute_reward(self) -> float:
        """Compute shaped reward."""
        left_tcp = self.data.site_xpos[self.left_tcp_id]
        right_tcp = self.data.site_xpos[self.right_tcp_id]
        left_obj = self.data.xpos[self.left_object_id]
        right_obj = self.data.xpos[self.right_object_id]

        reward = 0.0

        if self.reward_type == "sparse":
            # Only reward when both placed
            if self.left_placed and self.right_placed:
                reward = 1.0

        elif self.reward_type == "dense":
            # Left arm reward
            if not self.left_placed:
                if not self.left_grasped:
                    # Encourage reaching object
                    dist_to_obj = np.linalg.norm(left_tcp - left_obj)
                    reward -= 1.0 * dist_to_obj
                else:
                    # Encourage moving object to target
                    dist_to_target = np.linalg.norm(left_obj - self.target_pos)
                    reward -= 2.0 * dist_to_target

            # Right arm reward
            if not self.right_placed:
                if not self.right_grasped:
                    dist_to_obj = np.linalg.norm(right_tcp - right_obj)
                    reward -= 1.0 * dist_to_obj
                else:
                    dist_to_target = np.linalg.norm(right_obj - self.target_pos)
                    reward -= 2.0 * dist_to_target

            # Success bonus
            if self.left_placed:
                reward += 50.0
            if self.right_placed:
                reward += 50.0

        return reward

    def _check_success(self) -> bool:
        """Check if both objects placed successfully."""
        left_obj = self.data.xpos[self.left_object_id]
        right_obj = self.data.xpos[self.right_object_id]

        left_dist = np.linalg.norm(left_obj - self.target_pos)
        right_dist = np.linalg.norm(right_obj - self.target_pos)

        if left_dist < self.success_threshold and not self.left_placed:
            self.left_placed = True
        if right_dist < self.success_threshold and not self.right_placed:
            self.right_placed = True

        return self.left_placed and self.right_placed

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Set robot arms to low ready pose near table height (~0.5m)
        # Left arm joints: qpos[10:17]
        self.data.qpos[10] = 0.5      # joint1 (shoulder pan) - turn slightly forward
        self.data.qpos[11] = -0.8     # joint2 (shoulder lift) - NEGATIVE lowers arm
        self.data.qpos[12] = 0.0      # joint3 (shoulder roll)
        self.data.qpos[13] = -1.5     # joint4 (elbow) - bent
        self.data.qpos[14] = 0.0      # joint5 (wrist 1)
        self.data.qpos[15] = 0.0      # joint6 (wrist 2)
        self.data.qpos[16] = 0.0      # joint7 (wrist 3)

        # Right arm - mirror
        self.data.qpos[19] = -0.5     # joint1 (shoulder pan) - mirror
        self.data.qpos[20] = -0.8     # joint2 (shoulder lift) - NEGATIVE lowers arm
        self.data.qpos[21] = 0.0      # joint3 (shoulder roll)
        self.data.qpos[22] = -1.5     # joint4 (elbow) - bent
        self.data.qpos[23] = 0.0      # joint5 (wrist 1)
        self.data.qpos[24] = 0.0      # joint6 (wrist 2)
        self.data.qpos[25] = 0.0      # joint7 (wrist 3)

        # Randomize object positions in source bin
        source_pos = self.data.xpos[self.source_bin_id].copy()

        # Left object: left side of source bin
        left_pos = source_pos + np.array([
            np.random.uniform(-0.15, -0.05),
            np.random.uniform(0.05, 0.15),
            0.18  # Raised to sit on table surface (桌面+物体半高)
        ])
        self.data.qpos[self.left_obj_qpos:self.left_obj_qpos+3] = left_pos

        # Right object: right side of source bin
        right_pos = source_pos + np.array([
            np.random.uniform(-0.15, -0.05),
            np.random.uniform(-0.15, -0.05),
            0.18  # Raised to sit on table surface
        ])
        self.data.qpos[self.right_obj_qpos:self.right_obj_qpos+3] = right_pos

        # Reset orientation (quaternion)
        self.data.qpos[self.left_obj_qpos+3:self.left_obj_qpos+7] = [1, 0, 0, 0]
        self.data.qpos[self.right_obj_qpos+3:self.right_obj_qpos+7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        # Reset state
        self.current_step = 0
        self.left_grasped = False
        self.right_grasped = False
        self.left_placed = False
        self.right_placed = False

        obs = self._get_obs()

        # Add target positions to info for expert policy
        target_bin_pos = self.data.xpos[self.target_bin_id].copy()
        info = {
            "episode_success": False,
            "target_left": target_bin_pos + np.array([-0.1, 0.1, 0.15]),  # Left side of target bin
            "target_right": target_bin_pos + np.array([-0.1, -0.1, 0.15]),  # Right side of target bin
        }

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        # Parse action: [14 joint deltas, 2 gripper commands]
        joint_deltas = action[:14]
        gripper_cmds = action[14:16]

        # Apply joint position changes (clipped)
        self.data.qpos[:14] += np.clip(joint_deltas, -0.1, 0.1)

        # Gripper logic (simplified: threshold at 0.5)
        if gripper_cmds[0] > 0.5:
            self._attempt_grasp("left")
        else:
            self._release_grasp("left")

        if gripper_cmds[1] > 0.5:
            self._attempt_grasp("right")
        else:
            self._release_grasp("right")

        # Step physics
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        # Get observation and reward
        obs = self._get_obs()
        reward = self._compute_reward()
        success = self._check_success()

        # Termination conditions
        terminated = success
        truncated = self.current_step >= self.max_steps

        # Compute distances for debugging
        left_tcp = self.data.site_xpos[self.left_tcp_id]
        right_tcp = self.data.site_xpos[self.right_tcp_id]
        left_obj = self.data.xpos[self.left_object_id]
        right_obj = self.data.xpos[self.right_object_id]
        target_bin_pos = self.data.xpos[self.target_bin_id]

        info = {
            "episode_success": success,
            "left_placed": self.left_placed,
            "right_placed": self.right_placed,
            "episode_length": self.current_step,
            "left_distance": np.linalg.norm(left_tcp - left_obj) if not self.left_placed else 0.0,
            "right_distance": np.linalg.norm(right_tcp - right_obj) if not self.right_placed else 0.0,
        }

        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()

        return obs, reward, terminated, truncated, info

    def _attempt_grasp(self, arm: str):
        """Attempt to grasp object if close enough."""
        if arm == "left" and not self.left_grasped:
            tcp_pos = self.data.site_xpos[self.left_tcp_id]
            obj_pos = self.data.xpos[self.left_object_id]
            if np.linalg.norm(tcp_pos - obj_pos) < 0.08:  # 8cm grasp threshold
                self.left_grasped = True
        elif arm == "right" and not self.right_grasped:
            tcp_pos = self.data.site_xpos[self.right_tcp_id]
            obj_pos = self.data.xpos[self.right_object_id]
            if np.linalg.norm(tcp_pos - obj_pos) < 0.08:
                self.right_grasped = True

    def _release_grasp(self, arm: str):
        """Release grasped object."""
        if arm == "left":
            self.left_grasped = False
        elif arm == "right":
            self.right_grasped = False

    def render(self):
        """Render environment."""
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # TODO: Implement offscreen rendering
            raise NotImplementedError("rgb_array rendering not yet implemented")

    def close(self):
        """Clean up resources."""
        if self.viewer:
            self.viewer.close()


if __name__ == "__main__":
    # Quick test
    env = DualArmPickPlaceEnv(render_mode=None)

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step reward: {reward:.2f}, Terminated: {terminated}")

        if terminated or truncated:
            break

    env.close()
    print("Environment test complete!")
