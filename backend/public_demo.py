"""Robot-configured public pick-and-place demonstration.

This module is the runnable public path.  It deliberately keeps task policy
separate from the IK backend: the robot model and all joint/TCP names come from
``robot/robot_config.yaml``; :class:`ik_backend.MinkIKSolver` only sees the
loaded MuJoCo model plus that description.

The demonstration is kinematic.  It is intended for deterministic dataset and
video generation, and avoids depending on private actuator tuning or gripper
contact geometry.  A production controller can reuse the same IK solver and
replace the interpolation/carry policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import mujoco
import numpy as np

try:
    from .ik_backend import MinkIKSolver, kinematics_from_adapter
    from .robot_adapter import RobotAdapter
except ImportError:  # Direct script/module execution from backend/.
    from ik_backend import MinkIKSolver, kinematics_from_adapter
    from robot_adapter import RobotAdapter


FrameCallback = Callable[[mujoco.MjModel, mujoco.MjData, str], None]


@dataclass(frozen=True)
class DemoResult:
    items: tuple[str, ...]
    phases: tuple[str, ...]


class PublicPickPlaceDemo:
    """Small model-driven pick/place policy for the public scene."""

    def __init__(self, config_path: Optional[str] = None, side: str = "left"):
        repo = Path(__file__).resolve().parent.parent
        config = Path(config_path) if config_path else repo / "robot" / "robot_config.yaml"
        self.repo = repo
        self.adapter = RobotAdapter(str(config))
        scene_rel = self.adapter.get_asset_path("scene") or "scene/combined_scene.xml"
        scene_path = Path(scene_rel)

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.side = side
        self.arm_joint_names = self.adapter.get_arm_joints(side)
        self.arm_qpos = np.asarray(
            [self._joint_qpos(name) for name in self.arm_joint_names], dtype=int)
        self.tcp_name = self.adapter.get_tcp_site(side)
        self.tcp_id = self._site_id(self.tcp_name)
        mocap_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_mocap")
        self.grasp_mocap_index = (
            int(self.model.body_mocapid[mocap_body]) if mocap_body >= 0 else -1)
        self.base_joint = self.adapter.get_base_root_joint()
        self.base_qpos = self._joint_qpos(self.base_joint) if self.base_joint else None
        self.gripper_joint_names = self.adapter.get_gripper_joints(side)
        self.gripper_qpos = np.asarray(
            [self._joint_qpos(name) for name in self.gripper_joint_names], dtype=int)
        self.gripper_open, self.gripper_closed = self.adapter.get_gripper_range(side)
        self.solver = MinkIKSolver(
            self.model, kinematics_from_adapter(self.adapter, side))
        self._carried_body: Optional[int] = None
        self._carry_offset = np.zeros(3)
        self._phases: list[str] = []
        self._arm_body_ids = self._discover_arm_body_ids()
        self._arm_geom_ids = self._collision_geoms(self._arm_body_ids)
        self._collision_checks = 0
        self._minimum_clearance = float("inf")
        self._minimum_forbidden_clearance = float("inf")

    def _joint_id(self, name: Optional[str]) -> int:
        if not name:
            raise ValueError("joint name is required")
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint not found in public model: {name}")
        return int(jid)

    def _joint_qpos(self, name: Optional[str]) -> int:
        return int(self.model.jnt_qposadr[self._joint_id(name)])

    def _site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"site not found in public model: {name}")
        return int(sid)

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"body not found in public scene: {name}")
        return int(bid)

    def _body_freejoint_qpos(self, body_id: int) -> int:
        jid = int(self.model.body_jntadr[body_id])
        if (jid < 0 or int(self.model.jnt_type[jid])
                != int(mujoco.mjtJoint.mjJNT_FREE)):
            raise ValueError(f"body is not free-moving: {self.model.body(body_id).name}")
        return int(self.model.jnt_qposadr[jid])

    def _discover_arm_body_ids(self) -> set[int]:
        """Derive the controlled arm subtree from configured joint bodies."""
        roots = {
            int(self.model.jnt_bodyid[self._joint_id(name)])
            for name in self.arm_joint_names
        }
        bodies = set()
        for bid in range(1, self.model.nbody):
            current = bid
            while current > 0:
                if current in roots:
                    bodies.add(bid)
                    break
                current = int(self.model.body_parentid[current])
        return bodies

    def _collision_geoms(self, body_ids: set[int]) -> list[int]:
        return [
            gid for gid in range(self.model.ngeom)
            if int(self.model.geom_bodyid[gid]) in body_ids
            and (int(self.model.geom_contype[gid]) != 0
                 or int(self.model.geom_conaffinity[gid]) != 0)
        ]

    def _ik_collision_pairs(self, allowed_body_ids: set[int]):
        obstacle_bodies = {
            bid for bid in range(self.model.nbody)
            if bid not in self._arm_body_ids and bid not in allowed_body_ids
        }
        obstacles = self._collision_geoms(obstacle_bodies)
        return [
            (self._arm_geom_ids, obstacles),
            (self._arm_geom_ids, self._arm_geom_ids),
        ]

    def discover_items(self) -> list[str]:
        items = []
        for jid in range(self.model.njnt):
            if int(self.model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            name = self.model.body(int(self.model.jnt_bodyid[jid])).name
            if name != "base_link":
                items.append(name)
        return sorted(items)

    @property
    def collision_checks(self) -> int:
        """Number of interpolated trajectory samples validated in the run."""
        return self._collision_checks

    @property
    def minimum_forbidden_clearance(self) -> float:
        """Smallest signed contact distance among contacts that are not allowed."""
        return self._minimum_forbidden_clearance

    def _set_position_controls(self) -> None:
        """Match public position actuators to the current configured pose."""
        for side in ("left", "right"):
            for name in self.adapter.get_arm_joints(side):
                aid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                    self.adapter.get_actuator_name(name))
                self.data.ctrl[aid] = self.data.qpos[self._joint_qpos(name)]
            grip_aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                self.adapter.grippers[side]["actuator"])
            opened, _ = self.adapter.get_gripper_range(side)
            self.data.ctrl[grip_aid] = opened
        torso_name = self.adapter.config.get("torso", {}).get("joint")
        if torso_name:
            aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                self.adapter.get_actuator_name(torso_name))
            self.data.ctrl[aid] = self.data.qpos[self._joint_qpos(torso_name)]

    def _settle_items(self, callback: Optional[FrameCallback],
                      max_steps: int = 900) -> None:
        """Let randomized SKUs fall into the source bin under MuJoCo gravity."""
        item_dofs = []
        for name in self.discover_items():
            body_id = self._body_id(name)
            jid = int(self.model.body_jntadr[body_id])
            dof = int(self.model.jnt_dofadr[jid])
            item_dofs.extend(range(dof, dof + 6))

        if not item_dofs:
            mujoco.mj_forward(self.model, self.data)
            return

        self._set_position_controls()
        mujoco.mj_forward(self.model, self.data)
        for step in range(max_steps):
            mujoco.mj_step(self.model, self.data)
            if callback is not None and step % 8 == 0:
                callback(self.model, self.data, "settling")
            if step > 250 and np.max(np.abs(self.data.qvel[item_dofs])) < 0.015:
                break

        # Preserve the settled object poses, then put the robot back at its
        # exact configured carry pose so the IK seed is reproducible.
        settled = {}
        for name in self.discover_items():
            qadr = self._body_freejoint_qpos(self._body_id(name))
            settled[name] = self.data.qpos[qadr:qadr + 7].copy()
        self.data.qvel[:] = 0.0
        if self.base_qpos is not None:
            self.data.qpos[self.base_qpos:self.base_qpos + 3] = [0.64, 0.0, 0.0]
            self.data.qpos[self.base_qpos + 3:self.base_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        for side in ("left", "right"):
            for name, value in zip(
                    self.adapter.get_arm_joints(side),
                    self.adapter.get_rest_pose(side)):
                self.data.qpos[self._joint_qpos(name)] = value
            opened, _ = self.adapter.get_gripper_range(side)
            for name in self.adapter.get_gripper_joints(side):
                self.data.qpos[self._joint_qpos(name)] = opened
        torso_name = self.adapter.config.get("torso", {}).get("joint")
        if torso_name:
            self.data.qpos[self._joint_qpos(torso_name)] = float(
                self.adapter.rest_pose.get("torso", 0.0))
        for name, pose in settled.items():
            qadr = self._body_freejoint_qpos(self._body_id(name))
            self.data.qpos[qadr:qadr + 7] = pose
        self.data.time = 0.0
        self._set_position_controls()
        mujoco.mj_forward(self.model, self.data)

    def reset(self, *, randomize: bool = True, seed: Optional[int] = None,
              settle: bool = True,
              callback: Optional[FrameCallback] = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if self.base_qpos is not None:
            # Keep the chassis just outside the table edge (table xmin=0.95,
            # chassis half-length=0.30) while leaving both bins reachable.
            self.data.qpos[self.base_qpos:self.base_qpos + 3] = [0.64, 0.0, 0.0]
            self.data.qpos[self.base_qpos + 3:self.base_qpos + 7] = [1.0, 0.0, 0.0, 0.0]

        torso_cfg = self.adapter.config.get("torso", {})
        torso_name = torso_cfg.get("joint")
        if torso_name:
            self.data.qpos[self._joint_qpos(torso_name)] = float(
                self.adapter.rest_pose.get("torso", 0.0))

        for arm_side in ("left", "right"):
            for name, value in zip(
                    self.adapter.get_arm_joints(arm_side),
                    self.adapter.get_rest_pose(arm_side)):
                self.data.qpos[self._joint_qpos(name)] = value
        self._set_gripper(self.gripper_open)

        if randomize:
            rng = np.random.default_rng(seed)
            for index, name in enumerate(self.discover_items()):
                bid = self._body_id(name)
                qadr = self._body_freejoint_qpos(bid)
                if name.startswith("kebike"):
                    # Its proxy is a long cylinder whose body origin is at one
                    # end.  Align it roughly with the bin's long axis so the
                    # randomized drop begins fully inside the walls.
                    xy = [rng.uniform(1.240, 1.245), rng.uniform(0.17, 0.18)]
                    yaw = rng.uniform(-0.08, 0.08)
                else:
                    xy = [rng.uniform(1.120, 1.125), rng.uniform(0.34, 0.36)]
                    yaw = rng.uniform(-np.pi, np.pi)
                self.data.qpos[qadr:qadr + 3] = [
                    xy[0], xy[1], rng.uniform(0.76, 0.80)]
                self.data.qpos[qadr + 3:qadr + 7] = [
                    np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

        self._carried_body = None
        self._phases = []
        self._collision_checks = 0
        self._minimum_clearance = float("inf")
        self._minimum_forbidden_clearance = float("inf")
        mujoco.mj_forward(self.model, self.data)
        self._sync_grasp_anchor()
        mujoco.mj_forward(self.model, self.data)
        if settle:
            self._settle_items(callback)

    def _set_gripper(self, value: float) -> None:
        for qadr in self.gripper_qpos:
            self.data.qpos[qadr] = value

    def _update_carried_item(self) -> None:
        if self._carried_body is None:
            return
        qadr = self._body_freejoint_qpos(self._carried_body)
        self.data.qpos[qadr:qadr + 3] = (
            self.data.site_xpos[self.tcp_id] + self._carry_offset)

    def _sync_grasp_anchor(self) -> None:
        """Keep the visible weld/mocap marker on the configured TCP."""
        if self.grasp_mocap_index < 0:
            return
        self.data.mocap_pos[self.grasp_mocap_index] = (
            self.data.site_xpos[self.tcp_id])
        mujoco.mju_mat2Quat(
            self.data.mocap_quat[self.grasp_mocap_index],
            self.data.site_xmat[self.tcp_id].reshape(9))

    def _emit(self, phase: str, callback: Optional[FrameCallback]) -> None:
        mujoco.mj_forward(self.model, self.data)
        self._sync_grasp_anchor()
        self._update_carried_item()
        mujoco.mj_forward(self.model, self.data)
        if not self._phases or self._phases[-1] != phase:
            self._phases.append(phase)
        if callback is not None:
            callback(self.model, self.data, phase)

    @staticmethod
    def _display_name(value: Optional[str], fallback: str) -> str:
        return value if value else fallback

    def _check_collisions(
            self, phase: str, allowed_arm_body_ids: set[int],
            allowed_carried_body_ids: Optional[set[int]] = None,
            penetration_tolerance: float = 1e-3) -> None:
        """Reject penetrations at every interpolated trajectory sample.

        The grasped item's contact with the fingers is expected.  Contacts
        between the arm and everything else, arm self-collisions, and a
        carried item's contact with the environment are rejected.
        """
        allowed_carried = allowed_carried_body_ids or set()
        self._collision_checks += 1
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            distance = float(contact.dist)
            self._minimum_clearance = min(self._minimum_clearance, distance)
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            arm1, arm2 = body1 in self._arm_body_ids, body2 in self._arm_body_ids

            forbidden = False
            if arm1 and arm2:
                forbidden = True
            elif arm1:
                forbidden = body2 not in allowed_arm_body_ids
            elif arm2:
                forbidden = body1 not in allowed_arm_body_ids
            elif self._carried_body is not None:
                if body1 == self._carried_body:
                    forbidden = body2 not in allowed_carried
                elif body2 == self._carried_body:
                    forbidden = body1 not in allowed_carried

            if forbidden:
                self._minimum_forbidden_clearance = min(
                    self._minimum_forbidden_clearance, distance)
            if forbidden and distance < -penetration_tolerance:
                name1 = self._display_name(
                    self.model.geom(geom1).name, f"geom#{geom1}")
                name2 = self._display_name(
                    self.model.geom(geom2).name, f"geom#{geom2}")
                raise RuntimeError(
                    f"collision during {phase}: {name1} <-> {name2}, "
                    f"penetration={-1000.0 * distance:.2f} mm")

    def _interpolate(self, indices: np.ndarray, target: np.ndarray, frames: int,
                     phase: str, callback: Optional[FrameCallback],
                     allowed_body_ids: Optional[set[int]] = None,
                     allowed_carried_body_ids: Optional[set[int]] = None) -> None:
        start = self.data.qpos[indices].copy()
        allowed = allowed_body_ids or set()
        # Collision sampling is based on joint displacement, independently of
        # the requested video frame count.  This prevents a short animation
        # from jumping over a thin obstacle between two rendered frames.
        max_delta = float(np.max(np.abs(target - start))) if len(indices) else 0.0
        samples = max(2, frames, int(np.ceil(max_delta / 0.02)) + 1)
        emit_steps = set(np.linspace(1, samples - 1, max(1, frames - 1), dtype=int))
        for step, alpha in enumerate(np.linspace(0.0, 1.0, samples)[1:], start=1):
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            self.data.qpos[indices] = (1.0 - smooth) * start + smooth * target
            self._emit(phase, callback if step in emit_steps else None)
            self._check_collisions(
                phase, allowed, allowed_carried_body_ids)

    def _move_tcp(self, target: np.ndarray, frames: int, phase: str,
                  callback: Optional[FrameCallback],
                  allowed_body_ids: Optional[set[int]] = None,
                  allowed_carried_body_ids: Optional[set[int]] = None,
                  target_quat: Optional[np.ndarray] = None) -> None:
        allowed = allowed_body_ids or set()
        start_pos = self.data.site_xpos[self.tcp_id].copy()
        start_quat = np.empty(4)
        mujoco.mju_mat2Quat(
            start_quat, self.data.site_xmat[self.tcp_id].reshape(9))
        end_quat = start_quat if target_quat is None else np.asarray(
            target_quat, dtype=float)
        end_quat = end_quat / np.linalg.norm(end_quat)
        if np.dot(start_quat, end_quat) < 0.0:
            end_quat = -end_quat
        angle = 2.0 * np.arccos(np.clip(np.dot(start_quat, end_quat), -1.0, 1.0))
        segments = max(
            1, int(np.ceil(np.linalg.norm(target - start_pos) / 0.04)),
            int(np.ceil(angle / 0.15)))
        segment_frames = max(2, int(np.ceil(frames / segments)) + 1)

        # Resolve a chain of nearby Cartesian waypoints.  Collision-aware IK
        # only constrains its result; these short segments also constrain the
        # path between successive results.
        for step in range(1, segments + 1):
            alpha = step / segments
            waypoint = (1.0 - alpha) * start_pos + alpha * target
            quat = (1.0 - alpha) * start_quat + alpha * end_quat
            quat /= np.linalg.norm(quat)
            q_target, error = self.solver.solve(
                self.data, waypoint,
                target_quat=quat if target_quat is not None else None,
                max_iters=500,
                collision_pairs=self._ik_collision_pairs(allowed),
                collision_clearance=0.006,
                collision_detection_distance=0.10,
            )
            if not np.isfinite(error) or error > 0.02:
                raise RuntimeError(
                    f"IK failed for {phase}: error={error:.4f}, "
                    f"waypoint={step}/{segments}, target={waypoint.tolist()}")
            self._interpolate(
                self.arm_qpos, q_target[self.arm_qpos], segment_frames,
                phase, callback, allowed, allowed_carried_body_ids)

    def _move_base_y(self, target_y: float, frames: int, phase: str,
                     callback: Optional[FrameCallback],
                     allowed_body_ids: Optional[set[int]] = None,
                     allowed_carried_body_ids: Optional[set[int]] = None) -> None:
        if self.base_qpos is None:
            return
        indices = np.asarray([self.base_qpos + 1], dtype=int)
        self._interpolate(
            indices, np.asarray([target_y]), frames, phase, callback,
            allowed_body_ids, allowed_carried_body_ids)

    def _animate_gripper(self, target: float, frames: int, phase: str,
                         callback: Optional[FrameCallback],
                         allowed_body_ids: Optional[set[int]] = None) -> None:
        if len(self.gripper_qpos):
            self._interpolate(
                self.gripper_qpos, np.full(len(self.gripper_qpos), target),
                frames, phase, callback, allowed_body_ids)

    def _attach(self, body_id: int) -> None:
        """Preserve the contact pose reached by the physical fingertip TCP."""
        self._carried_body = body_id
        self._carry_offset = (
            self.data.xpos[body_id] - self.data.site_xpos[self.tcp_id])

    def _detach_at(self, position: np.ndarray) -> None:
        if self._carried_body is None:
            return
        qadr = self._body_freejoint_qpos(self._carried_body)
        self.data.qpos[qadr:qadr + 3] = position
        self._carried_body = None
        mujoco.mj_forward(self.model, self.data)

    def run(self, items: Optional[Iterable[str]] = None, *,
            randomize: bool = True, seed: Optional[int] = None,
            callback: Optional[FrameCallback] = None) -> DemoResult:
        self.reset(
            randomize=randomize, seed=seed, settle=True, callback=callback)
        selected = self.discover_items()
        if items is not None:
            requested = set(items)
            selected = [name for name in selected if name in requested]
        if not selected:
            raise ValueError("no matching items in public scene")

        self._emit("initial", callback)
        source_crate = self._body_id("crate_0")
        destination_crate = self._body_id("crate_2")
        # This simplified public wrist cannot remain exactly vertical at the
        # deepest bin pose without its coarse link-5/link-7 proxies overlap.
        # Leave wrist roll free and let the collision limit choose a clear
        # posture at each Cartesian waypoint.
        grasp_quat = None
        for name in selected:
            body_id = self._body_id(name)
            target_allowed = {body_id}
            body_pos = self.data.xpos[body_id].copy()
            proxy_geoms = self._collision_geoms({body_id})
            if not proxy_geoms:
                raise RuntimeError(f"item has no collision geometry: {name}")
            item_pos = self.data.geom_xpos[proxy_geoms[0]].copy()
            grasp_height = 0.06
            grasp = item_pos + np.array([0.0, 0.0, grasp_height])
            # The long fingers can grasp slightly off-center while the wrist
            # and finger backs stay inside the bin's safe opening.
            grasp[0] = np.clip(grasp[0], 1.15, 1.25)
            grasp[1] = np.clip(grasp[1], 0.28, 0.34)
            pregrasp = grasp.copy()
            pregrasp[2] = max(pregrasp[2] + 0.22, 0.98)
            self._move_base_y(0.0, 12, f"{name}:base_pick", callback)
            self._interpolate(
                self.arm_qpos, np.asarray(self.adapter.get_rest_pose(self.side)),
                20, f"{name}:tuck", callback, target_allowed)
            safe_entry = self.data.site_xpos[self.tcp_id].copy()
            safe_entry[2] = max(safe_entry[2], 1.18)
            self._move_tcp(
                safe_entry, 18, f"{name}:raise_clear", callback,
                target_allowed)
            self._move_tcp(
                pregrasp, 24, f"{name}:pregrasp", callback,
                target_allowed, target_quat=grasp_quat)
            self._move_tcp(
                grasp, 24, f"{name}:approach_pick", callback,
                target_allowed, target_quat=grasp_quat)
            self._animate_gripper(
                self.gripper_closed, 8, f"{name}:close", callback,
                target_allowed)
            self._attach(body_id)
            self._move_tcp(
                pregrasp, 18, f"{name}:lift", callback, target_allowed,
                {source_crate}, grasp_quat)
            self._move_base_y(
                -0.55, 28, f"{name}:transfer", callback, target_allowed)

            # Place collision-proxy centers in separate target slots.  This
            # accounts for meshes such as kebike whose body origin is at one
            # end rather than at the physical center.
            proxy_destination = np.array([
                1.25 if name.startswith("kebike") else 1.13,
                -0.35,
            ])
            destination = body_pos.copy()
            destination[:2] += proxy_destination - item_pos[:2]
            place = destination - self._carry_offset
            hover = place + np.array([0.0, 0.0, 0.27])
            hover[2] = max(hover[2], 1.18)
            self._move_tcp(
                hover, 24, f"{name}:approach_place", callback,
                target_allowed, target_quat=grasp_quat)
            self._move_tcp(
                place, 18, f"{name}:descend_place", callback,
                target_allowed, {destination_crate}, grasp_quat)
            self._detach_at(destination)
            self._animate_gripper(
                self.gripper_open, 8, f"{name}:open", callback,
                target_allowed)
            self._move_tcp(
                hover, 16, f"{name}:retreat", callback,
                allowed_body_ids=target_allowed,
                target_quat=grasp_quat)

        self._emit("final", callback)
        return DemoResult(tuple(selected), tuple(self._phases))
