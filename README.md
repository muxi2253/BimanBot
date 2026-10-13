# 双臂移动机器人 MuJoCo 开源示例

这是一个基于 MuJoCo、Gymnasium 和 Mink 的双臂移动操作示例。仓库内附可直接加载的公开 MJCF、参数化机器人适配层、模型驱动 IK、Gym 环境，以及 GIF 和 LeRobot 数据导出工具。

## 效果展示

> **注意**: 演示 GIF 和数据集文件较大（约 300 MB），未包含在代码仓库中。
> 
> 📦 **获取方式**:
> - [从 GitHub Releases 下载](../../releases)（推荐）
> - [查看完整下载指南](docs/DATASET_DOWNLOAD.md)
> - 或运行 `SHOWCASE_RELEASE.md` 中的命令自行生成

### 完整工作区视角
机器人从左侧料箱取出两个物体，移动底盘后放入右侧料箱。

<!-- 下载后放置在: recordings/showcase_public/demo_wide.gif -->
![Demo Wide View](https://via.placeholder.com/800x450.png?text=Demo+Wide+View+-+See+Releases)

### TCP 跟随特写
相机跟随左臂末端，展示接近、闭合、抬升、转运、下降和释放的完整流程。

<!-- 下载后放置在: recordings/showcase_public/demo_closeup.gif -->
![Demo Closeup View](https://via.placeholder.com/800x450.png?text=Demo+Closeup+-+See+Releases)

### 双臂 Reach 环境
Gym 环境中的周期性双臂到达动作。

<!-- 下载后放置在: recordings/showcase_public/reach_episode.gif -->
![Reach Episode](https://via.placeholder.com/800x450.png?text=Reach+Episode+-+See+Releases)

### 空间融合实验
四种非学习深度先验基线的对比。

<!-- 下载后放置在: recordings/showcase_public/spatial_fusion.mp4 -->

## 当前可用功能

| 功能 | 状态 | 入口 |
|---|---:|---|
| 公开双臂移动机器人 MJCF | 可用 | `robot/dual_arm_robot.xml` |
| 配置驱动机器人适配层 | 可用 | `backend/robot_adapter.py` |
| Mink 单臂位置/姿态 IK | 可用 | `backend/ik_backend.py` |
| 双物体拾取、搬运、放置演示 | 可用 | `backend/public_demo.py` |
| 双臂 Reach Gym 环境 | 可用 | `backend/gym_env_reach.py` |
| Pick-place Gym 环境、三路 RGB 观察 | 可用 | `backend/gym_env.py` |
| 摩擦与关节阻尼随机化 | 可用 | `DualArmPickPlaceEnv(domain_randomization=True)` |
| qpos 轨迹转 LeRobot v2.1 | 可用 | `backend/serialize_to_lerobot.py` |
| VR/手柄遥操作、训练脚本 | 尚未提供 | — |
| 深度图与分割图录制 | 尚未提供 | — |

公开 pick-place 演示使用 IK 与平滑插值，并以运动学方式携带物体。它适合验证运动学、相机和录制链路；它不是夹爪接触动力学基准。

## 录制命令

macOS 的 MuJoCo 离屏渲染需要使用 `mjpython`：

```bash
mjpython tools/record_demo.py
mjpython tools/record_demo.py --items kebike_0 --seed 42 --fps 20
mjpython tools/record_gym_episode.py --steps 100 --fps 25
```

SKU 默认从料箱上方随机生成并在重力下沉降。使用 `--seed` 可复现布局，使用 `--no-rand` 可退回 XML 中的固定位置。Linux 可直接把 `mjpython` 换成 `python`。

## 安装与快速验证

```bash
conda create -n dual-arm python=3.10 -y
conda activate dual-arm
pip install -r requirements.txt

python examples/basic_demo.py
python backend/gym_env_reach.py
python -m pytest
```

查看模型：

```bash
# macOS（交互式 viewer 需要 mjpython）
mjpython tools/view_robot.py

# Linux / Windows
python tools/view_robot.py
```

该入口会读取 `robot_config.yaml`，将双臂初始化到镜像的胸前收臂姿态，
并同步 position actuator；直接打开裸 MJCF 不会自动应用这份配置姿态。

## IK 是否依赖具体机器人的 URDF？

IK 求解必须有一个**具体的运动学模型**，否则没有关节轴、连杆变换、关节限位和 TCP 可供求解。但 `MinkIKSolver` 不依赖某个厂商 URDF，也没有写死当前机器人的关节名或自由度：

```text
URDF（可选的建模来源） ─┐
                        ├─> MuJoCo MjModel + robot_config.yaml
手写/转换后的 MJCF ─────┘                │
                                         v
                         RobotKinematics -> MinkIKSolver
```

Mink 在运行时直接使用已经加载的 `mujoco.MjModel`。URDF 只是生成 MJCF 的一种来源，并非 IK API 的必需参数。更换机器人时需要：

- 一份能被 MuJoCo 加载的 MJCF；
- 在 `robot_config.yaml` 中填写每条手臂的关节、TCP site、夹爪、可选底盘关节与静止姿态；
- 场景中的相机挂载和物体资产。

`MinkIKSolver` 会从模型枚举非目标手臂关节、自由底盘和物体 free joint，并在默认情况下将它们固定。只有显式传入 `allow_base_motion=True` 时，配置的底盘关节才参与求解。

## 最小 IK 示例

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

## Gym 与 LeRobot

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

LeRobot 转换器读取 `exports/episode_XXXXXX/trajectory_qpos.npy` 和 `annotation.json`，回放三路 RGB 相机并写出 Parquet、MP4 与 v2.1 元数据：

```bash
pip install pandas pyarrow "imageio[ffmpeg]"
mjpython backend/serialize_to_lerobot.py --src exports --out data/lerobot --fps 10
```

当前仓库提供转换器，不包含遥操作采集器。输入轨迹的第二维必须与当前场景的 `model.nq` 一致。

## 公开数据集

> **注意**: 数据集文件较大，未包含在代码仓库中。请从以下方式获取：
> - **GitHub Releases**: [下载数据集压缩包](../../releases)（推荐）
> - **自行生成**: 运行 `SHOWCASE_RELEASE.md` 中的命令生成数据

### LeRobot v2.1 格式数据集
**路径**: `recordings/showcase_public/lerobot/`  
**大小**: 约 4.6 MB

包含内容：
- **Parquet 数据表**: 488 帧的关节状态、夹爪位置、TCP 状态和动作
- **三路相机视频**（10 fps，480×640）：
  - `observation.image.top`: 俯视工作区视角
  - `observation.image.left_wrist`: 左臂腕部相机
  - `observation.image.right_wrist`: 右臂腕部相机
- **元数据**: 包含特征维度、数据类型、分割信息等

特征说明：
- `observation.state.joint`: 14 个关节位置（双臂各 7 个关节）
- `observation.gripper_position`: 2 个夹爪开度
- `observation.state.end`: 12 维 TCP 位姿（双臂各 6 维：xyz + 欧拉角）
- `actions`: 16 维下一步动作（14 关节 + 2 夹爪）

### 原始导出数据
**路径**: `recordings/showcase_public/exports/episode_000000/`  
**大小**: 约 187 MB

包含 488 帧的完整传感器数据：
- **RGB 图像** (`.npy`): 三路相机的原始 RGB 数组
- **深度图** (`.npy`): 米制深度值
- **分割掩码** (`.npy`): 整数标签图
- **点云** (`.xyz`): 从俯视相机反投影的世界坐标点云
- **可视化** (`.png`): RGB、深度和分割的 PNG 预览图
- **轨迹**: `trajectory_qpos.npy` 和 `annotation.json`

数据采集说明：物体从料箱上方随机掉落，经重力沉降后稳定，然后开始录制。初始状态保存在 `initial_qpos.npy`。

## 目录

```text
robot/                  公开 URDF、MJCF 与 robot_config.yaml
scene/                  料箱、物体和组合场景
backend/ik_backend.py   与机器人命名解耦的 Mink IK
backend/public_demo.py  可录制的公开 pick-place 演示
backend/gym_env*.py     Reach 与 pick-place Gym 环境
backend/camera_config.py 配置驱动的三相机注入
tools/record_*.py       GIF 录制脚本
tests/                  公开链路回归测试
```

## License

[MIT](LICENSE)
