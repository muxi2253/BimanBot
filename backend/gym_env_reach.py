#!/usr/bin/env python3
"""
Simplified Gymnasium Environment for Dual-Arm Robot
Task: Reach target positions with both TCP sites

This is a minimal working example for RL/IL research.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))
try:
    from .robot_adapter import RobotAdapter, load_default_adapter
except ImportError:  # Direct script execution.
    from robot_adapter import RobotAdapter, load_default_adapter


class DualArmReachEnv(gym.Env):
    """
    Simple reaching task for dual-arm robot

    Observation: joint positions (14D) + TCP positions (6D) = 20D
    Action: joint velocity commands (14D)
    Reward: negative distance to target positions
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=500, config_path=None):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        # Resolve the model relative to the selected relocatable asset config.
        self.adapter = (
            RobotAdapter(config_path) if config_path else load_default_adapter())
        robot_path = self.adapter.get_asset_path('model')
        if robot_path is None:
            raise ValueError("robot config does not define assets.model")
        self.model = mujoco.MjModel.from_xml_path(str(robot_path))
        self.data = mujoco.MjData(self.model)

        # Get joint indices
        self.left_joints = self.adapter.get_arm_joints('left')
        self.right_joints = self.adapter.get_arm_joints('right')
        self.all_joints = self.left_joints + self.right_joints

        self.joint_indices = []
        self.joint_qpos_addrs = []
        self.joint_qvel_addrs = []

        for jname in self.all_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self.joint_indices.append(jid)
            self.joint_qpos_addrs.append(self.model.jnt_qposadr[jid])
            self.joint_qvel_addrs.append(self.model.jnt_dofadr[jid])

        # Get TCP site indices
        self.left_tcp_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.adapter.get_tcp_site('left'))
        self.right_tcp_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.adapter.get_tcp_site('right'))

        # Get actuator indices (position servos for arms)
        self.actuator_indices = []
        for jname in self.all_joints:
            act_name = self.adapter.get_actuator_name(jname)
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            self.actuator_indices.append(act_id)

        # Define spaces
        # Observation: joint pos (14) + TCP pos (6)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32
        )

        # Action: joint position deltas (14)
        self.action_space = spaces.Box(
            low=-0.1,
            high=0.1,
            shape=(14,),
            dtype=np.float32
        )

        # Target positions (will be randomized in reset)
        self.target_left = np.array([0.3, 0.3, 1.2])
        self.target_right = np.array([0.3, -0.3, 1.2])

        # Renderer
        self.renderer = None
        if self.render_mode == "human":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def _get_obs(self):
        """Get current observation"""
        # Joint positions
        joint_pos = np.array([self.data.qpos[addr] for addr in self.joint_qpos_addrs])

        # TCP positions
        left_tcp_pos = self.data.site_xpos[self.left_tcp_id].copy()
        right_tcp_pos = self.data.site_xpos[self.right_tcp_id].copy()

        obs = np.concatenate([joint_pos, left_tcp_pos, right_tcp_pos])
        return obs.astype(np.float32)

    def _get_info(self):
        """Get additional info"""
        left_tcp_pos = self.data.site_xpos[self.left_tcp_id]
        right_tcp_pos = self.data.site_xpos[self.right_tcp_id]

        left_dist = np.linalg.norm(left_tcp_pos - self.target_left)
        right_dist = np.linalg.norm(right_tcp_pos - self.target_right)

        return {
            'left_distance': float(left_dist),
            'right_distance': float(right_dist),
            'total_distance': float(left_dist + right_dist),
            'success': (left_dist < 0.05 and right_dist < 0.05)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set base height
        self.data.qpos[2] = 0.0

        # Set arms to rest pose with small noise
        left_rest = self.adapter.get_rest_pose('left')
        right_rest = self.adapter.get_rest_pose('right')

        if self.np_random is not None:
            left_rest += self.np_random.normal(0, 0.1, size=left_rest.shape)
            right_rest += self.np_random.normal(0, 0.1, size=right_rest.shape)

        # Apply to qpos
        for i, addr in enumerate(self.joint_qpos_addrs[:7]):
            self.data.qpos[addr] = left_rest[i]
        for i, addr in enumerate(self.joint_qpos_addrs[7:]):
            self.data.qpos[addr] = right_rest[i]

        for addr, act_id in zip(self.joint_qpos_addrs, self.actuator_indices):
            self.data.ctrl[act_id] = self.data.qpos[addr]

        torso = self.adapter.config.get('torso', {})
        if torso:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, torso['joint'])
            aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                self.adapter.get_actuator_name(torso['joint']))
            torso_q = float(self.adapter.rest_pose.get('torso', 0.0))
            self.data.qpos[self.model.jnt_qposadr[jid]] = torso_q
            self.data.ctrl[aid] = torso_q

        for side in ('left', 'right'):
            opened, _ = self.adapter.get_gripper_range(side)
            for name in self.adapter.get_gripper_joints(side):
                jid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.data.qpos[self.model.jnt_qposadr[jid]] = opened
            aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                self.adapter.grippers[side]['actuator'])
            self.data.ctrl[aid] = opened

        # Randomize target positions
        if self.np_random is not None:
            self.target_left = np.array([
                self.np_random.uniform(0.2, 0.5),
                self.np_random.uniform(0.1, 0.4),
                self.np_random.uniform(1.0, 1.5)
            ])
            self.target_right = np.array([
                self.np_random.uniform(0.2, 0.5),
                self.np_random.uniform(-0.4, -0.1),
                self.np_random.uniform(1.0, 1.5)
            ])

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one step"""
        # Apply action as position delta
        current_qpos = np.array([self.data.qpos[addr] for addr in self.joint_qpos_addrs])
        target_qpos = current_qpos + action

        # Clip to joint limits
        for i, jid in enumerate(self.joint_indices):
            if self.model.jnt_limited[jid]:
                jnt_range = self.model.jnt_range[jid]
                target_qpos[i] = np.clip(target_qpos[i], jnt_range[0], jnt_range[1])

        # Set actuator controls
        for i, act_id in enumerate(self.actuator_indices):
            self.data.ctrl[act_id] = target_qpos[i]

        # Step simulation (10 substeps for smooth control)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        # Calculate reward
        info = self._get_info()
        reward = -info['total_distance']

        # Success bonus
        if info['success']:
            reward += 10.0

        # Check termination
        terminated = info['success']
        truncated = self.current_step >= self.max_steps

        observation = self._get_obs()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)

            self.renderer.update_scene(self.data)
            return self.renderer.render()

        elif self.render_mode == "human":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)

            self.renderer.update_scene(self.data)
            pixels = self.renderer.render()

            # Simple display (requires additional packages for proper window)
            return pixels

        return None

    def close(self):
        """Clean up"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# Register environment
gym.register(
    id='DualArmReach-v0',
    entry_point='backend.gym_env_reach:DualArmReachEnv',
    max_episode_steps=500,
)


if __name__ == '__main__':
    # Simple test
    print("Testing DualArmReach environment...")

    env = DualArmReachEnv()
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial distance: {info['total_distance']:.3f}m")

    # Run a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, dist={info['total_distance']:.3f}m")

        if terminated or truncated:
            break

    env.close()
    print("\nEnvironment test complete!")
