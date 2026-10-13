# 数据集下载指南 / Dataset Download Guide

[English](#english) | [中文](#中文)

---

## 中文

### 概述

由于演示 GIF 和数据集文件较大（总计约 300 MB），它们未包含在主代码仓库中。本文档说明如何获取这些文件。

### 文件清单

| 文件 | 大小 | 说明 |
|---|---:|---|
| `demo_wide.gif` | 25 MB | 完整工作区视角演示 |
| `demo_closeup.gif` | 57 MB | TCP 跟随特写演示 |
| `reach_episode.gif` | 15 MB | 双臂 Reach 环境演示 |
| `spatial_fusion.mp4` | 64 KB | 空间融合实验视频 |
| `lerobot/` 数据集 | 4.6 MB | LeRobot v2.1 格式数据集 |
| `exports/` 数据集 | 187 MB | 原始导出数据（RGB、深度、分割、点云） |

**总大小**: 约 289 MB

### 方法一：从 GitHub Releases 下载（推荐）

1. 访问仓库的 [Releases 页面](../../releases)
2. 下载最新版本的 `showcase-assets.zip`
3. 解压到项目根目录：
   ```bash
   cd /path/to/BimanBot
   unzip showcase-assets.zip
   ```
4. 验证文件结构：
   ```bash
   ls -lh recordings/showcase_public/*.gif
   ls -lh recordings/showcase_public/lerobot/
   ls -lh recordings/showcase_public/exports/
   ```

### 方法二：自行生成

如果你已经配置好环境，可以运行以下命令重新生成所有演示和数据集：

```bash
# 1. 生成演示 GIF（约 5-10 分钟）
mjpython tools/record_demo.py --seed 50 --out recordings/showcase_public
mjpython tools/record_gym_episode.py --steps 140 --output recordings/showcase_public/reach_episode.gif

# 2. 生成空间融合实验（约 10-15 分钟）
python tools/spatial_fusion_experiment.py --samples 20 --seed 7 --out recordings/showcase_public/spatial_fusion

# 3. 导出原始数据（约 2-3 分钟）
python tools/export_public_data.py --out recordings/showcase_public/exports

# 4. 转换为 LeRobot 格式（约 1-2 分钟）
pip install pandas pyarrow "imageio[ffmpeg]"
mjpython backend/serialize_to_lerobot.py \
  --src recordings/showcase_public/exports \
  --out recordings/showcase_public/lerobot --fps 10 --overwrite
```

**注意**: macOS 需要使用 `mjpython` 进行离屏渲染，Linux 可以直接使用 `python`。

### 方法三：从外部存储获取

如果 GitHub Releases 下载速度慢，可以从以下备用链接获取：

- 百度网盘: *待添加*
- Google Drive: *待添加*
- 阿里云盘: *待添加*

### 验证数据完整性

下载或生成后，运行以下命令验证文件完整性：

```bash
python tools/check_release_assets.py
```

该脚本会检查所有必需文件是否存在且格式正确。

---

## English

### Overview

Due to their large size (approximately 300 MB total), demo GIFs and dataset files are not included in the main code repository. This document explains how to obtain these files.

### File List

| File | Size | Description |
|---|---:|---|
| `demo_wide.gif` | 25 MB | Full workspace view demo |
| `demo_closeup.gif` | 57 MB | TCP-following closeup demo |
| `reach_episode.gif` | 15 MB | Dual-arm reach environment demo |
| `spatial_fusion.mp4` | 64 KB | Spatial fusion experiment video |
| `lerobot/` dataset | 4.6 MB | LeRobot v2.1 format dataset |
| `exports/` dataset | 187 MB | Raw export data (RGB, depth, segmentation, point clouds) |

**Total size**: ~289 MB

### Method 1: Download from GitHub Releases (Recommended)

1. Visit the repository's [Releases page](../../releases)
2. Download the latest `showcase-assets.zip`
3. Extract to project root:
   ```bash
   cd /path/to/BimanBot
   unzip showcase-assets.zip
   ```
4. Verify file structure:
   ```bash
   ls -lh recordings/showcase_public/*.gif
   ls -lh recordings/showcase_public/lerobot/
   ls -lh recordings/showcase_public/exports/
   ```

### Method 2: Generate Yourself

If you have the environment configured, regenerate all demos and datasets:

```bash
# 1. Generate demo GIFs (~5-10 minutes)
mjpython tools/record_demo.py --seed 50 --out recordings/showcase_public
mjpython tools/record_gym_episode.py --steps 140 --output recordings/showcase_public/reach_episode.gif

# 2. Generate spatial fusion experiment (~10-15 minutes)
python tools/spatial_fusion_experiment.py --samples 20 --seed 7 --out recordings/showcase_public/spatial_fusion

# 3. Export raw data (~2-3 minutes)
python tools/export_public_data.py --out recordings/showcase_public/exports

# 4. Convert to LeRobot format (~1-2 minutes)
pip install pandas pyarrow "imageio[ffmpeg]"
mjpython backend/serialize_to_lerobot.py \
  --src recordings/showcase_public/exports \
  --out recordings/showcase_public/lerobot --fps 10 --overwrite
```

**Note**: macOS requires `mjpython` for offscreen rendering; Linux can use `python` directly.

### Method 3: External Storage

If GitHub Releases download is slow, use these alternative links:

- Baidu Netdisk: *TBA*
- Google Drive: *TBA*
- Aliyun Drive: *TBA*

### Verify Data Integrity

After downloading or generating, verify file integrity:

```bash
python tools/check_release_assets.py
```

This script checks that all required files exist and are properly formatted.
