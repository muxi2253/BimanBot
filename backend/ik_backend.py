"""Model-driven MuJoCo IK backend.

The public simulation uses this interface instead of a robot-specific IK class.
Joint and TCP names come from :class:`RobotKinematics`; no URDF path, vendor
name, torso layout, or fixed degree-of-freedom count is embedded here.

Mink is the default backend because it operates directly on the loaded MuJoCo
model and can therefore use either a public MJCF or a private deployment
asset. Pinocchio can be added behind the same interface when a deployment
needs its own model-based dynamics or analytical constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import mujoco
import numpy as np

if TYPE_CHECKING:
    from robot_adapter import RobotAdapter


@dataclass(frozen=True)
class RobotKinematics:
    """Names needed by the IK layer for one or more manipulators."""

    arm_joint_names: tuple[str, ...]
    tcp_site_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...] = ()
    movable_base_joint: Optional[str] = None
    rest_positions: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.arm_joint_names:
            raise ValueError("arm_joint_names must not be empty")
        if len(self.tcp_site_names) == 0:
            raise ValueError("tcp_site_names must not be empty")
        if self.rest_positions and len(self.rest_positions) != len(self.arm_joint_names):
            raise ValueError("rest_positions must match arm_joint_names")


def _joint_qpos_addresses(model: mujoco.MjModel, names: Iterable[str]) -> np.ndarray:
    addresses = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint not found in model: {name}")
        joint_type = int(model.jnt_type[jid])
        if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                              int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError(f"IK joint must be hinge or slide: {name}")
        addresses.append(int(model.jnt_qposadr[jid]))
    return np.asarray(addresses, dtype=int)


def _joint_dof_addresses(model: mujoco.MjModel, names: Iterable[str]) -> np.ndarray:
    addresses = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint not found in model: {name}")
        addresses.append(int(model.jnt_dofadr[jid]))
    return np.asarray(addresses, dtype=int)


def _joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name is None:
        raise ValueError(
            f"joint {joint_id} has no name; model-driven IK requires named joints")
    return name


class MinkIKSolver:
    """Small, model-driven position/orientation IK wrapper around Mink."""

    def __init__(self, model: mujoco.MjModel, robot: RobotKinematics,
                 *, solver: str = "quadprog", dt: float = 1e-2,
                 damping: float = 1e-2) -> None:
        try:
            import mink
        except ImportError as exc:  # pragma: no cover - depends on deployment
            raise RuntimeError("Mink is required for MinkIKSolver") from exc

        self._mink = mink
        self.model = model
        self.robot = robot
        self.solver = solver
        self.dt = float(dt)
        self.damping = float(damping)
        self.arm_qpos = _joint_qpos_addresses(model, robot.arm_joint_names)
        self.arm_dof = _joint_dof_addresses(model, robot.arm_joint_names)
        self._tcp_ids = []
        for name in robot.tcp_site_names:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"TCP site not found in model: {name}")
            self._tcp_ids.append(sid)
        self._config = mink.Configuration(model)
        self._configuration_limit = mink.ConfigurationLimit(model)

        # Only the configured arm may move. All unrelated scalar joints and
        # free joints are derived from the loaded model and pinned by name, so
        # this layer does not need to know a vendor torso, wheel, neck, object,
        # or gripper layout.
        arm_names = set(robot.arm_joint_names)
        scalar_pins = {}
        free_joint_names = []
        for jid in range(model.njnt):
            name = _joint_name(model, jid)
            joint_type = int(model.jnt_type[jid])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                free_joint_names.append(name)
            elif joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                int(mujoco.mjtJoint.mjJNT_SLIDE)) and name not in arm_names:
                scalar_pins[name] = 1e-6

        self._scalar_pin_limit = (
            mink.VelocityLimit(model, scalar_pins) if scalar_pins else None)
        self._free_pin_limits = {
            name: mink.FreeJointVelocityLimit(
                model, [1e-6] * 3, [1e-6] * 3, joint_name=name)
            for name in free_joint_names
        }
        if (robot.movable_base_joint is not None
                and robot.movable_base_joint not in self._free_pin_limits):
            raise ValueError(
                f"movable base joint is not a free joint: {robot.movable_base_joint}")
        self._mobile_base_limit = None
        if robot.movable_base_joint is not None:
            self._mobile_base_limit = mink.FreeJointVelocityLimit(
                model, [0.25, 0.25, 1e-6], [1e-6, 1e-6, 0.5],
                joint_name=robot.movable_base_joint)

    def solve(self, data: mujoco.MjData, target_pos: Sequence[float], *,
              tcp_index: int = 0, target_quat: Optional[Sequence[float]] = None,
              seed_qpos: Optional[Sequence[float]] = None,
              max_iters: int = 80,
              allow_base_motion: bool = False,
              collision_pairs: Optional[
                  Sequence[tuple[Sequence[int | str], Sequence[int | str]]]
              ] = None,
              collision_clearance: float = 0.01,
              collision_detection_distance: float = 0.08,
              ) -> tuple[np.ndarray, float]:
        """Return a full qpos target and final task error."""
        if not 0 <= tcp_index < len(self._tcp_ids):
            raise IndexError("tcp_index is outside tcp_site_names")
        q0 = data.qpos.copy()
        if seed_qpos is not None:
            seed = np.asarray(seed_qpos, dtype=float)
            if seed.shape == data.qpos.shape:
                q0 = seed.copy()
            elif seed.shape == self.arm_qpos.shape:
                q0[self.arm_qpos] = seed
            else:
                raise ValueError(
                    "seed_qpos must be a full qpos vector or one value per arm joint")
        self._config.update(q0)
        task = self._mink.FrameTask(
            self.robot.tcp_site_names[tcp_index], "site",
            position_cost=1.0,
            orientation_cost=0.5 if target_quat is not None else 0.0,
            lm_damping=self.damping,
        )
        if target_quat is None:
            rotation = self._mink.SO3.from_matrix(
                self._config.data.site_xmat[self._tcp_ids[tcp_index]].reshape(3, 3).copy())
        else:
            rotation = self._mink.SO3(np.asarray(target_quat, dtype=float))
        task.set_target(self._mink.SE3.from_rotation_and_translation(
            rotation, np.asarray(target_pos, dtype=float)))

        limits = [self._configuration_limit]
        if self._scalar_pin_limit is not None:
            limits.append(self._scalar_pin_limit)
        for name, limit in self._free_pin_limits.items():
            if allow_base_motion and name == self.robot.movable_base_joint:
                continue
            limits.append(limit)
        if allow_base_motion and self._mobile_base_limit is not None:
            limits.append(self._mobile_base_limit)
        if collision_pairs:
            limits.append(self._mink.CollisionAvoidanceLimit(
                self.model,
                collision_pairs,
                gain=0.6,
                minimum_distance_from_collisions=collision_clearance,
                collision_detection_distance=collision_detection_distance,
                bound_relaxation=1e-4,
            ))

        error = float("inf")
        for _ in range(max_iters):
            try:
                velocity = self._mink.solve_ik(
                    self._config, [task], self.dt, self.solver,
                    damping=self.damping, limits=limits)
            except self._mink.exceptions.NoSolutionFound:
                return self._config.q.copy(), float("inf")
            self._config.integrate_inplace(velocity, self.dt)
            task_error = task.compute_error(self._config)
            error = float(np.linalg.norm(task_error[:3]))
            if target_quat is not None:
                error = max(error, float(np.linalg.norm(task_error[3:])))
            if error < 1e-3:
                break
        return self._config.q.copy(), error


def make_robot_kinematics(*, arm_joint_names: Sequence[str],
                          tcp_site_names: Sequence[str],
                          gripper_joint_names: Sequence[str] = (),
                          movable_base_joint: Optional[str] = None,
                          rest_positions: Sequence[float] = ()) -> RobotKinematics:
    """Construct configuration from an asset adapter or user config file."""
    return RobotKinematics(
        tuple(arm_joint_names), tuple(tcp_site_names),
        tuple(gripper_joint_names), movable_base_joint, tuple(rest_positions))


def kinematics_from_adapter(adapter: "RobotAdapter", side: str) -> RobotKinematics:
    """Build an IK description from the public robot adapter."""
    return make_robot_kinematics(
        arm_joint_names=adapter.get_arm_joints(side),
        tcp_site_names=[adapter.get_tcp_site(side)],
        gripper_joint_names=adapter.get_gripper_joints(side),
        movable_base_joint=adapter.get_base_root_joint(),
        rest_positions=adapter.get_rest_pose(side),
    )
