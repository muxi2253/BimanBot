"""gym_env.py - Gymnasium-compatible RL environment for dual-arm manipulation.

This environment wraps the MuJoCo simulation to provide a standard Gym interface
for reinforcement learning and imitation learning algorithms.

Features:
- Dual-arm mobile robot with 14 DOF arms + 2 grippers
- Multi-view RGB observations (top, left_wrist, right_wrist)
- Continuous action space (joint positions + gripper commands)
- Customizable reward functions (sparse/dense/shaped)
- Domain randomization support

Example:
    import gymnasium as gym
    from backend.gym_env import DualArmPickPlaceEnv

    env = gym.make('DualArmPickPlace-v0')
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)

try:
    from .camera_config import CAMERA_CONFIG, RENDER_H, RENDER_W, load_scene_with_cameras
    from .robot_adapter import RobotAdapter, load_default_adapter
except ImportError:  # Direct script/module execution from backend/.
    from camera_config import CAMERA_CONFIG, RENDER_H, RENDER_W, load_scene_with_cameras
    from robot_adapter import RobotAdapter, load_default_adapter


def _body_id(model, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"body not found: {name}")
    return int(body_id)


def _body_pose(data, body_id):
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


class DualArmPickPlaceEnv(gym.Env):
    """Dual-arm mobile robot pick-and-place environment.

    Observation Space:
        Dict(
            joint: Box(14) - left/right arm joint angles (rad)
            gripper: Box(2) - left/right total gripper opening (m)
            tcp_pose: Box(12) - left/right TCP xyz + euler_xyz
            image_top: Box(H,W,3) - top camera RGB (uint8)
            image_left: Box(H,W,3) - left wrist camera RGB (uint8)
            image_right: Box(H,W,3) - right wrist camera RGB (uint8)
        )

    Action Space:
        Box(16) - bounded joint deltas (14) + gripper commands (2)

    Reward:
        - sparse: +1 if object in target box, else 0
        - dense: -distance(object, target) with shaping
        - shaped: dense + grasp bonus + penalty for collisions
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene_path: Optional[str] = None,
        config_path: Optional[str] = None,
        reward_type: str = "sparse",
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
        camera_obs: bool = True,
        domain_randomization: bool = False,
    ):
        """Initialize the environment.

        Args:
            scene_path: Path to MJCF scene file (default: combined_scene.xml)
            config_path: Robot asset config. Relative assets resolve from it.
            reward_type: "sparse", "dense", or "shaped"
            max_episode_steps: Maximum steps per episode
            render_mode: "human" for GUI, "rgb_array" for offscreen
            camera_obs: Include camera observations in obs space
            domain_randomization: Enable domain randomization
        """
        super().__init__()

        self.reward_type = reward_type
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.camera_obs = camera_obs
        self.domain_randomization = domain_randomization
        self.adapter = (
            RobotAdapter(config_path) if config_path else load_default_adapter())

        # Load scene
        if scene_path is None:
            scene_path = self.adapter.get_asset_path("scene")
        if scene_path is None:
            raise ValueError("robot config does not define assets.scene")

        if camera_obs:
            self.model, self.data = load_scene_with_cameras(
                self.adapter.get_camera_mounts(), self.adapter)
        else:
            self.model = mujoco.MjModel.from_xml_path(scene_path)
            self.data = mujoco.MjData(self.model)

        # Build joint indices
        self._build_joint_indices()
        self._base_friction = self.model.geom_friction.copy()
        self._base_damping = self.model.dof_damping.copy()

        # Define observation space
        gripper_high = np.asarray([
            2.0 * max(self.adapter.get_gripper_range(side))
            for side in ("left", "right")
        ], dtype=np.float32)
        obs_spaces = {
            "joint": spaces.Box(-np.pi, np.pi, shape=(14,), dtype=np.float32),
            "gripper": spaces.Box(
                np.zeros(2, dtype=np.float32), gripper_high, dtype=np.float32),
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32),
        }

        if camera_obs:
            for cam in ["top", "left_wrist", "right_wrist"]:
                obs_spaces[f"image_{cam}"] = spaces.Box(
                    0, 255, shape=(RENDER_H, RENDER_W, 3), dtype=np.uint8
                )

        self.observation_space = spaces.Dict(obs_spaces)

        # Define action space: 14 joint angles + 2 gripper commands
        self.action_space = spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)

        # Renderer for visual observations
        self.renderer = None
        if camera_obs or render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, RENDER_H, RENDER_W)

        # Viewer for human rendering
        self.viewer = None

        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0

        # Target item and goal position (set in reset)
        self.target_item = None
        self.target_goal = None

    def _build_joint_indices(self):
        """Build indices for joint positions and actuators."""
        def jqpos(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return int(self.model.jnt_qposadr[jid]) if jid >= 0 else None

        def sid(name):
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            return int(i) if i >= 0 else None

        def aid(name):
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            return int(i) if i >= 0 else None

        left_arm = self.adapter.get_arm_joints("left")
        right_arm = self.adapter.get_arm_joints("right")
        left_grippers = self.adapter.get_gripper_joints("left")
        right_grippers = self.adapter.get_gripper_joints("right")
        self.joint_indices = {
            "left_arm": [jqpos(name) for name in left_arm],
            "right_arm": [jqpos(name) for name in right_arm],
            "left_gripper": [jqpos(name) for name in left_grippers],
            "right_gripper": [jqpos(name) for name in right_grippers],
            "left_tcp": sid(self.adapter.get_tcp_site("left")),
            "right_tcp": sid(self.adapter.get_tcp_site("right")),
        }

        self.actuator_indices = {
            "left_arm": [aid(self.adapter.get_actuator_name(name)) for name in left_arm],
            "right_arm": [aid(self.adapter.get_actuator_name(name)) for name in right_arm],
            "left_gripper": aid(self.adapter.grippers["left"]["actuator"]),
            "right_gripper": aid(self.adapter.grippers["right"]["actuator"]),
        }

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Joint positions
        left_joints = np.array([
            self.data.qpos[i] for i in self.joint_indices["left_arm"]
        ], dtype=np.float32)
        right_joints = np.array([
            self.data.qpos[i] for i in self.joint_indices["right_arm"]
        ], dtype=np.float32)
        joint_obs = np.concatenate([left_joints, right_joints])

        left_grip = 2.0 * self.data.qpos[self.joint_indices["left_gripper"][0]]
        right_grip = 2.0 * self.data.qpos[self.joint_indices["right_gripper"][0]]
        gripper_obs = np.array([left_grip, right_grip], dtype=np.float32)

        # TCP poses
        from scipy.spatial.transform import Rotation

        left_tcp_pos = self.data.site_xpos[self.joint_indices["left_tcp"]]
        left_tcp_mat = self.data.site_xmat[self.joint_indices["left_tcp"]].reshape(3, 3)
        left_tcp_euler = Rotation.from_matrix(left_tcp_mat).as_euler("xyz")

        right_tcp_pos = self.data.site_xpos[self.joint_indices["right_tcp"]]
        right_tcp_mat = self.data.site_xmat[self.joint_indices["right_tcp"]].reshape(3, 3)
        right_tcp_euler = Rotation.from_matrix(right_tcp_mat).as_euler("xyz")

        tcp_obs = np.concatenate([
            left_tcp_pos, left_tcp_euler,
            right_tcp_pos, right_tcp_euler
        ]).astype(np.float32)

        obs = {
            "joint": joint_obs,
            "gripper": gripper_obs,
            "tcp_pose": tcp_obs,
        }

        # Camera observations
        if self.camera_obs and self.renderer:
            for cam in ["top", "left_wrist", "right_wrist"]:
                self.renderer.update_scene(self.data, camera=cam)
                obs[f"image_{cam}"] = self.renderer.render().copy()

        return obs

    def _compute_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        """Compute reward, termination, and info dict.

        Returns:
            reward: scalar reward
            terminated: episode should end
            info: additional information
        """
        info = {}

        if self.target_item is None:
            return 0.0, False, info

        # Get object position
        item_body_id = _body_id(self.model, self.target_item)
        item_pos, _ = _body_pose(self.data, item_body_id)

        # Check if in goal region
        goal_center = np.array(self.target_goal[:3])
        goal_tolerance = 0.15  # 15cm radius
        distance = np.linalg.norm(item_pos - goal_center)

        in_goal = distance < goal_tolerance
        info["distance_to_goal"] = float(distance)
        info["success"] = in_goal

        if self.reward_type == "sparse":
            reward = 1.0 if in_goal else 0.0

        elif self.reward_type == "dense":
            # Negative distance + bonus for reaching goal
            reward = -distance
            if in_goal:
                reward += 10.0

        elif self.reward_type == "shaped":
            # Dense reward + additional shaping
            reward = -distance

            # Grasp bonus (check if object is lifted)
            if item_pos[2] > 0.1:  # 10cm above ground
                reward += 1.0
                info["is_lifted"] = True

            # Goal bonus
            if in_goal:
                reward += 10.0

            # Collision penalty
            n_contacts = self.data.ncon
            if n_contacts > 5:  # Threshold for "too many contacts"
                reward -= 0.1
                info["excessive_contacts"] = True

        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        # Termination condition
        terminated = in_goal or (self.current_step >= self.max_episode_steps)

        return reward, terminated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set robot to the configured public rest pose.
        REST_POSE = np.concatenate([
            self.adapter.get_rest_pose("left"),
            self.adapter.get_rest_pose("right"),
        ])

        for i, qpos_idx in enumerate(
            self.joint_indices["left_arm"] + self.joint_indices["right_arm"]
        ):
            self.data.qpos[qpos_idx] = REST_POSE[i]
            actuator = (self.actuator_indices["left_arm"]
                        + self.actuator_indices["right_arm"])[i]
            self.data.ctrl[actuator] = REST_POSE[i]

        # Open grippers
        left_open, _ = self.adapter.get_gripper_range("left")
        right_open, _ = self.adapter.get_gripper_range("right")
        self.data.qpos[self.joint_indices["left_gripper"]] = left_open
        self.data.qpos[self.joint_indices["right_gripper"]] = right_open
        self.data.ctrl[self.actuator_indices["left_gripper"]] = left_open
        self.data.ctrl[self.actuator_indices["right_gripper"]] = right_open

        torso = self.adapter.config.get("torso", {})
        if torso:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, torso["joint"])
            aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                self.adapter.get_actuator_name(torso["joint"]))
            torso_q = float(self.adapter.rest_pose.get("torso", 0.0))
            self.data.qpos[self.model.jnt_qposadr[jid]] = torso_q
            self.data.ctrl[aid] = torso_q

        base_name = self.adapter.get_base_root_joint()
        if base_name:
            base_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, base_name)
            base_qpos = int(self.model.jnt_qposadr[base_id])
            self.data.qpos[base_qpos:base_qpos + 3] = [0.60, 0.0, 0.0]
            self.data.qpos[base_qpos + 3:base_qpos + 7] = [1.0, 0.0, 0.0, 0.0]

        # Domain randomization
        if self.domain_randomization:
            self._apply_domain_randomization()

        # Randomize object position
        self._randomize_objects()
        self._settle_objects()
        if self.target_item is not None:
            target_body = _body_id(self.model, self.target_item)
            self.target_goal = [
                1.2, -0.35, float(self.data.xpos[target_body, 2]), 0.0]

        # Reset tracking
        self.current_step = 0
        self.episode_reward = 0.0

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: 16-dim array [joint_targets(14), gripper_cmds(2)]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Scale action from [-1, 1] to joint limits
        action = np.clip(action, -1.0, 1.0)

        # Map to actuator control
        joint_action = action[:14]
        gripper_action = action[14:16]

        # Apply bounded joint-position deltas. Jumping directly across a full
        # joint range in one 30 Hz action destabilizes position actuators.
        joint_names = (self.adapter.get_arm_joints("left")
                       + self.adapter.get_arm_joints("right"))
        joint_qpos = (self.joint_indices["left_arm"]
                      + self.joint_indices["right_arm"])
        for i, (name, qpos_idx, act_idx) in enumerate(zip(
                joint_names,
                joint_qpos,
                self.actuator_indices["left_arm"] + self.actuator_indices["right_arm"])):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo, hi = self.model.jnt_range[jid]
            target = self.data.qpos[qpos_idx] + 0.02 * joint_action[i]
            self.data.ctrl[act_idx] = np.clip(target, lo, hi)

        for side, value in zip(("left", "right"), gripper_action):
            opened, closed = self.adapter.get_gripper_range(side)
            command = closed + 0.5 * (value + 1.0) * (opened - closed)
            self.data.ctrl[self.actuator_indices[f"{side}_gripper"]] = command

        # Step simulation (control at 30Hz, sim at 500Hz)
        sim_steps = int(1.0 / (30 * self.model.opt.timestep))
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        reward, terminated, info = self._compute_reward()

        # Track episode
        self.current_step += 1
        self.episode_reward += reward

        truncated = self.current_step >= self.max_episode_steps

        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.current_step,
            }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            self.viewer.sync()

        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, RENDER_H, RENDER_W)
            self.renderer.update_scene(self.data, camera="top")
            return self.renderer.render()

    def close(self):
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()
        if self.viewer:
            self.viewer.close()

    def _randomize_objects(self):
        """Randomize object positions within crate."""
        # Find all free objects except the robot's floating base.
        items = []
        base_joint = self.adapter.get_base_root_joint()
        for jid in range(self.model.njnt):
            if int(self.model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            joint_name = self.model.joint(jid).name
            if joint_name == base_joint:
                continue
            items.append(self.model.body(int(self.model.jnt_bodyid[jid])).name)

        for item_index, item_name in enumerate(items):
            try:
                body_id = _body_id(self.model, item_name)
                # Randomize within crate bounds
                x = self.np_random.uniform(1.07, 1.27)
                lane_width = 0.34 / max(1, len(items))
                y0 = 0.18 + item_index * lane_width
                y = self.np_random.uniform(y0 + 0.02, y0 + lane_width - 0.02)
                z = self.np_random.uniform(0.96, 1.08)
                yaw = self.np_random.uniform(0, 2 * np.pi)

                # Set qpos (assuming freejoint)
                jnt_id = self.model.body_jntadr[body_id]
                qpos_addr = self.model.jnt_qposadr[jnt_id]

                self.data.qpos[qpos_addr:qpos_addr+3] = [x, y, z]
                # Set quaternion from yaw
                self.data.qpos[qpos_addr+3:qpos_addr+7] = [
                    np.cos(yaw/2), 0, 0, np.sin(yaw/2)
                ]
            except ValueError:
                continue

        # Set target (pick first available item)
        self.target_item = items[0] if items else None
        self.target_goal = [1.2, -0.35, 0.95, 0.0]

    def _settle_objects(self, max_steps: int = 900):
        """Drop randomized objects into the source bin before the episode."""
        base_joint = self.adapter.get_base_root_joint()
        object_joints = []
        object_dofs = []
        for jid in range(self.model.njnt):
            if int(self.model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            if self.model.joint(jid).name == base_joint:
                continue
            object_joints.append(jid)
            dof = int(self.model.jnt_dofadr[jid])
            object_dofs.extend(range(dof, dof + 6))

        if not object_dofs:
            mujoco.mj_forward(self.model, self.data)
            return

        # Save the configured robot pose. It may flex under gravity while the
        # SKUs fall, but the episode should begin at the exact carry pose.
        robot_qpos = self.data.qpos.copy()
        mujoco.mj_forward(self.model, self.data)
        for step in range(max_steps):
            mujoco.mj_step(self.model, self.data)
            if step > 250 and np.max(np.abs(self.data.qvel[object_dofs])) < 0.015:
                break

        object_poses = []
        for jid in object_joints:
            qadr = int(self.model.jnt_qposadr[jid])
            object_poses.append((qadr, self.data.qpos[qadr:qadr + 7].copy()))
        self.data.qpos[:] = robot_qpos
        for qadr, pose in object_poses:
            self.data.qpos[qadr:qadr + 7] = pose
        self.data.qvel[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _apply_domain_randomization(self):
        """Apply domain randomization to dynamics and visuals."""
        self.model.geom_friction[:] = self._base_friction
        self.model.dof_damping[:] = self._base_damping
        # Randomize friction
        for i in range(self.model.ngeom):
            self.model.geom_friction[i, 0] *= self.np_random.uniform(0.8, 1.2)

        # Randomize damping
        self.model.dof_damping[:] *= self.np_random.uniform(
            0.9, 1.1, size=self.model.nv)


# Register environment with Gymnasium
gym.register(
    id="DualArmPickPlace-v0",
    entry_point="backend.gym_env:DualArmPickPlaceEnv",
    max_episode_steps=500,
)
