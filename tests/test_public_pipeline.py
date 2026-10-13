import mujoco
import numpy as np

from gym_env import DualArmPickPlaceEnv
from ik_backend import MinkIKSolver, kinematics_from_adapter
from public_demo import PublicPickPlaceDemo
from robot_adapter import load_default_adapter


def test_configured_robot_names_exist_in_public_scene():
    adapter = load_default_adapter()
    model = mujoco.MjModel.from_xml_path(adapter.get_asset_path("scene"))

    for side in ("left", "right"):
        for name in adapter.get_arm_joints(side) + adapter.get_gripper_joints(side):
            assert mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, adapter.get_tcp_site(side)) >= 0


def test_model_driven_ik_moves_only_configured_arm():
    demo = PublicPickPlaceDemo(side="left")
    demo.reset(seed=0)
    before = demo.data.qpos.copy()
    tcp = demo.data.site_xpos[demo.tcp_id].copy()
    solver = MinkIKSolver(
        demo.model, kinematics_from_adapter(demo.adapter, "left"))

    target, error = solver.solve(demo.data, tcp + [0.015, 0.0, 0.0])
    assert error < 1e-3
    movable = set(demo.arm_qpos.tolist())
    pinned = np.array([i for i in range(demo.model.nq) if i not in movable])
    np.testing.assert_allclose(target[pinned], before[pinned], atol=2e-6)


def test_public_pick_place_finishes_in_destination_bin():
    demo = PublicPickPlaceDemo()
    item = demo.discover_items()[0]
    carry_distances = []

    def capture_carry_distance(model, data, phase):
        if demo._carried_body is None or not phase.endswith(("lift", "transfer")):
            return
        proxy = demo._collision_geoms({demo._carried_body})[0]
        carry_distances.append(float(np.linalg.norm(
            data.geom_xpos[proxy] - data.site_xpos[demo.tcp_id])))

    result = demo.run(
        items=[item], seed=0, callback=capture_carry_distance)
    position = demo.data.xpos[demo._body_id(item)]
    proxy = demo._collision_geoms({demo._body_id(item)})[0]
    proxy_position = demo.data.geom_xpos[proxy]

    assert result.items == (item,)
    np.testing.assert_allclose(proxy_position[:2], [1.13, -0.35], atol=1e-6)
    assert 0.50 < position[2] < 0.70
    assert demo.collision_checks > 100
    assert demo.minimum_forbidden_clearance >= -1e-3
    assert carry_distances and max(carry_distances) < 0.08
    np.testing.assert_allclose(
        demo.data.mocap_pos[demo.grasp_mocap_index],
        demo.data.site_xpos[demo.tcp_id], atol=1e-9)
    assert any("approach_pick" in phase for phase in result.phases)
    assert any("descend_place" in phase for phase in result.phases)


def test_randomized_items_settle_inside_source_bin():
    demo = PublicPickPlaceDemo()
    demo.reset(seed=42)
    positions = np.asarray([
        demo.data.xpos[demo._body_id(name)] for name in demo.discover_items()])

    assert np.all((positions[:, 0] > 0.95) & (positions[:, 0] < 1.45))
    assert np.all((positions[:, 1] > 0.16) & (positions[:, 1] < 0.54))
    assert np.all((positions[:, 2] > 0.50) & (positions[:, 2] < 0.70))
    assert demo.data.time == 0.0


def test_pick_place_gym_advances_without_numerical_reset():
    env = DualArmPickPlaceEnv(camera_obs=False)
    try:
        obs, _ = env.reset(seed=7)
        action = np.linspace(-0.25, 0.25, 16, dtype=np.float32)
        for _ in range(10):
            obs, _, _, _, _ = env.step(action)
        assert env.data.time > 0.30
        assert all(np.all(np.isfinite(value)) for value in obs.values())
    finally:
        env.close()
