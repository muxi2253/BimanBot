<div align="center">

# 🤖 双臂移动机器人 MuJoCo 开源示例

**基于 MuJoCo、Gymnasium 和 Mink 的双臂移动操作平台**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.0+-green.svg)](https://mujoco.org/)

[English](README_EN.md) | 简体中文

</div>

---

## ✨ 特性

- 🎯 **即用型 MJCF 模型** - 公开的双臂移动机器人完整模型
- 🔧 **配置驱动架构** - 参数化机器人适配层，易于更换机器人
- 🎮 **模型驱动 IK** - 基于 Mink 的位置/姿态逆运动学求解
- 🏋️ **Gymnasium 环境** - 标准化的 RL 训练环境（Reach & Pick-Place）
- 📹 **完整数据流** - GIF 录制、RGB-D-Seg 导出、LeRobot v2.1 转换
- 🔀 **域随机化** - 摩擦与关节阻尼随机化支持

---

## 🎬 效果展示

<table>
  <tr>
    <td width="50%" align="center">
      <b>完整工作区视角</b><br/>
      <sub>机器人从左侧料箱取出物体，移动底盘后放入右侧</sub><br/><br/>
      <img src="recordings/showcase_public/demo_wide.gif" width="100%"/>
    </td>
    <td width="50%" align="center">
      <b>TCP 跟随特写</b><br/>
      <sub>相机跟随左臂末端，展示抓取与放置流程</sub><br/><br/>
      <img src="recordings/showcase_public/demo_closeup.gif" width="100%"/>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <b>双臂 Reach 环境</b><br/>
      <sub>Gym 环境中的周期性双臂到达动作</sub><br/><br/>
      <img src="recordings/showcase_public/reach_episode.gif" width="100%"/>
    </td>
    <td width="50%" align="center">
      <b>空间融合实验</b><br/>
      <sub>四种非学习深度先验基线对比</sub><br/><br/>
      <a href="recordings/showcase_public/spatial_fusion.mp4">📹 观看视频</a>
    </td>
  </tr>
</table>

> **📦 数据集下载**: 完整数据集（约 192 MB）请查看 [数据集下载指南](docs/DATASET_DOWNLOAD.md)

---

## 🚀 快速开始

### 安装依赖

```bash
conda create -n dual-arm python=3.10 -y
conda activate dual-arm
pip install -r requirements.txt
```

### 运行示例

```bash
# 基础演示
python examples/basic_demo.py

# 双臂 Reach 环境
python backend/gym_env_reach.py

# 运行测试
python -m pytest
```

### 查看模型

```bash
# 只查看机器人（默认）
mjpython tools/view_robot.py

# 查看完整场景（包含桌子和料箱）
mjpython tools/view_robot.py --scene

# Linux / Windows
python tools/view_robot.py --scene
```

> 💡 该入口会读取 `robot_config.yaml`，将双臂初始化到镜像的胸前收臂姿态，并同步 position actuator。使用 `--scene` 参数可加载包含桌子和料箱的完整工作场景。

---

## 📋 功能列表

| 功能 | 入口 | 状态 |
|---|---|:---:|
| 公开双臂移动机器人 MJCF | [`robot/dual_arm_robot.xml`](robot/dual_arm_robot.xml) | ✅ |
| 配置驱动机器人适配层 | [`backend/robot_adapter.py`](backend/robot_adapter.py) | ✅ |
| Mink 单臂位置/姿态 IK | [`backend/ik_backend.py`](backend/ik_backend.py) | ✅ |
| 双物体拾取、搬运、放置演示 | [`backend/public_demo.py`](backend/public_demo.py) | ✅ |
| 双臂 Reach Gym 环境 | [`backend/gym_env_reach.py`](backend/gym_env_reach.py) | ✅ |
| Pick-place Gym 环境 + 三路 RGB | [`backend/gym_env.py`](backend/gym_env.py) | ✅ |
| 摩擦与关节阻尼随机化 | `domain_randomization=True` | ✅ |
| qpos 轨迹转 LeRobot v2.1 | [`backend/serialize_to_lerobot.py`](backend/serialize_to_lerobot.py) | ✅ |
| VR/手柄遥操作 | — | 🚧 |
| 训练脚本 | — | 🚧 |
| 深度图与分割图录制 | — | 🚧 |

> 💡 **说明**: 公开 pick-place 演示使用 IK 与平滑插值，并以运动学方式携带物体。适合验证运动学、相机和录制链路；不是夹爪接触动力学基准。

---

## 📹 录制命令

macOS 的 MuJoCo 离屏渲染需要使用 `mjpython`：

```bash
# 生成演示 GIF
mjpython tools/record_demo.py
mjpython tools/record_demo.py --items kebike_0 --seed 42 --fps 20

# 录制 Gym 环境
mjpython tools/record_gym_episode.py --steps 100 --fps 25
```

> 💡 **提示**: SKU 默认从料箱上方随机生成并在重力下沉降。使用 `--seed` 可复现布局，使用 `--no-rand` 可退回 XML 中的固定位置。Linux 可直接用 `python` 替换 `mjpython`。

---

## 🔧 IK 求解器

### 是否依赖具体机器人的 URDF？

IK 求解必须有一个**具体的运动学模型**，但 `MinkIKSolver` 不依赖某个厂商 URDF，也没有写死当前机器人的关节名或自由度：

```text
URDF（可选）─┐
            ├─> MuJoCo MjModel + robot_config.yaml
手写 MJCF ──┘            │
                        ▼
            RobotKinematics -> MinkIKSolver
```

**更换机器人时需要**：
- ✅ 一份能被 MuJoCo 加载的 MJCF
- ✅ 在 `robot_config.yaml` 中配置关节、TCP site、夹爪、底盘等
- ✅ 场景中的相机挂载和物体资产

> 💡 `MinkIKSolver` 会自动枚举并固定非目标关节。只有显式传入 `allow_base_motion=True` 时，底盘关节才参与求解。

### 最小 IK 示例

```python
import mujoco
from backend.ik_backend import MinkIKSolver, kinematics_from_adapter
from backend.robot_adapter import load_default_adapter

adapter = load_default_adapter()
model = mujoco.MjModel.from_xml_path(adapter.get_asset_path("scene"))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

solver = MinkIKSolver(model, kinematics_from_adapter(adapter, "left"))
qpos_target, error = solver.solve(data, [1.1, 0.3, 1.1])
```

---

## 🏋️ Gym 环境与 LeRobot

### Gym 环境使用

```python
from backend.gym_env import DualArmPickPlaceEnv

env = DualArmPickPlaceEnv(
    reward_type="shaped",
    camera_obs=True,
    domain_randomization=True,
)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

### LeRobot v2.1 转换

LeRobot 转换器读取轨迹和注释，回放三路 RGB 相机并写出 Parquet、MP4 与 v2.1 元数据：

```bash
pip install pandas pyarrow "imageio[ffmpeg]"
mjpython backend/serialize_to_lerobot.py --src exports --out data/lerobot --fps 10
```

> ⚠️ **注意**: 输入轨迹的第二维必须与当前场景的 `model.nq` 一致。当前仓库提供转换器，不包含遥操作采集器。

---

## 📦 公开数据集

> **注意**: 数据集文件较大，未包含在代码仓库中。请从以下方式获取：
> - **GitHub Releases**: [下载数据集压缩包](../../releases)（推荐）
> - **自行生成**: 运行 `SHOWCASE_RELEASE.md` 中的命令生成数据

### LeRobot v2.1 格式数据集

<details>
<summary><b>📊 数据集详情</b> (点击展开)</summary>

**路径**: `recordings/showcase_public/lerobot/`  
**大小**: 约 4.6 MB

**包含内容**：
- 🗂️ **Parquet 数据表**: 488 帧的关节状态、夹爪位置、TCP 状态和动作
- 📹 **三路相机视频**（10 fps，480×640）：
  - `observation.image.top`: 俯视工作区视角
  - `observation.image.left_wrist`: 左臂腕部相机
  - `observation.image.right_wrist`: 右臂腕部相机
- 📋 **元数据**: 包含特征维度、数据类型、分割信息等

**特征说明**：
- `observation.state.joint`: 14 个关节位置（双臂各 7 个关节）
- `observation.gripper_position`: 2 个夹爪开度
- `observation.state.end`: 12 维 TCP 位姿（双臂各 6 维：xyz + 欧拉角）
- `actions`: 16 维下一步动作（14 关节 + 2 夹爪）

</details>

### 原始导出数据

<details>
<summary><b>🔬 原始数据详情</b> (点击展开)</summary>

**路径**: `recordings/showcase_public/exports/episode_000000/`  
**大小**: 约 187 MB

**包含 488 帧的完整传感器数据**：
- 🖼️ **RGB 图像** (`.npy`): 三路相机的原始 RGB 数组
- 📏 **深度图** (`.npy`): 米制深度值
- 🎨 **分割掩码** (`.npy`): 整数标签图
- ☁️ **点云** (`.xyz`): 从俯视相机反投影的世界坐标点云
- 🖼️ **可视化** (`.png`): RGB、深度和分割的 PNG 预览图
- 📈 **轨迹**: `trajectory_qpos.npy` 和 `annotation.json`

> 📝 **数据采集说明**: 物体从料箱上方随机掉落，经重力沉降后稳定，然后开始录制。初始状态保存在 `initial_qpos.npy`。

</details>

---

## 📂 项目结构

```text
BimanBot/
├── 🤖 robot/              公开 URDF、MJCF 与 robot_config.yaml
├── 🎬 scene/              料箱、物体和组合场景
├── ⚙️  backend/
│   ├── ik_backend.py      与机器人命名解耦的 Mink IK
│   ├── public_demo.py     可录制的公开 pick-place 演示
│   ├── gym_env*.py        Reach 与 pick-place Gym 环境
│   └── camera_config.py   配置驱动的三相机注入
├── 🛠️  tools/
│   └── record_*.py        GIF 录制脚本
└── ✅ tests/              公开链路回归测试
```

---

## 📄 License

本项目采用 [MIT License](LICENSE) 开源。

---

<div align="center">

**⭐ 如果这个项目对你有帮助，欢迎给个 Star！**

[提交 Issue](../../issues) · [查看文档](docs/) · [加入讨论](../../discussions)

</div>
