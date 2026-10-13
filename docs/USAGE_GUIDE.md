# 使用指南

所有命令均从仓库根目录运行。

## 模型与初始姿态

基础动力学检查：

```bash
python examples/basic_demo.py
```

交互式查看：

```bash
# macOS
mjpython tools/view_robot.py

# Linux / Windows
python tools/view_robot.py
```

查看脚本从 `robot/robot_config.yaml` 读取胸前收臂姿态，并同步关节状态与
position actuator。按 `H` 可恢复该姿态。

## Robot Adapter 与 IK

```python
import mujoco
from backend.ik_backend import MinkIKSolver, kinematics_from_adapter
from backend.robot_adapter import load_default_adapter

adapter = load_default_adapter()
model = mujoco.MjModel.from_xml_path(adapter.get_asset_path("scene"))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

solver = MinkIKSolver(model, kinematics_from_adapter(adapter, "left"))
qpos_target, position_error = solver.solve(
    data,
    data.site(adapter.get_tcp_site("left")).xpos + [0.01, 0.0, 0.0],
)
```

IK 运行时依赖加载后的 MuJoCo 模型，不要求 URDF 参数，也不写死当前机器人的
关节名。更换机器人需要提供可加载的 MJCF 和匹配的 YAML 配置。

## Gym

Reach 示例：

```bash
python examples/gym_demo.py
```

Pick-place 环境：

```python
from backend.gym_env import DualArmPickPlaceEnv

env = DualArmPickPlaceEnv(
    reward_type="dense",
    camera_obs=True,
    domain_randomization=True,
)
obs, info = env.reset(seed=7)
obs, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

`camera_obs=True` 返回 `top`、`left_wrist`、`right_wrist` 三路 RGB 图像。

## 录制

```bash
# macOS
mjpython tools/record_demo.py --seed 42
mjpython tools/record_gym_episode.py --steps 100 --fps 25
```

详细参数见 [`GIF_RECORDING_GUIDE.md`](GIF_RECORDING_GUIDE.md)。

## LeRobot 文件导出

输入目录约定：

```text
exports/
  episode_000000/
    annotation.json
    trajectory_qpos.npy
    trajectory_meta.json      # 可选
```

执行转换：

```bash
mjpython backend/serialize_to_lerobot.py \
  --src exports \
  --out data/lerobot \
  --fps 10
```

当前转换器生成 LeRobot v2.1 文件。当前 LeRobot 0.6.1 使用 v3.0，不能直接加载
该输出；仓库也尚未提供公开 episode collector。

## 回归检查

```bash
python -m pytest -q
python tools/check_release_assets.py
```

当前测试覆盖公开主链路，不覆盖真实夹爪接触抓取、策略训练或 sim-to-real。

