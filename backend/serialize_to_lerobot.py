"""serialize_to_lerobot.py — 把 exports/ 目录转成 LeRobot v2.1 数据集格式。

输入（由任意采集器按下列约定生成）：
    exports/
      episode_XXXXXX/
        annotation.json             — dr_seed, success, item_result 等
        trajectory_qpos.npy         — (T, nq) 全 qpos 序列
        trajectory_meta.json        — 可选：fps, n_frames, nq, source

输出（LeRobot v2.1）：
    data/lerobot/
      data/chunk-000/episode_XXXXXX.parquet
      videos/chunk-000/observation.image.{top,left_wrist,right_wrist}/episode_XXXXXX.mp4
      meta/info.json / tasks.jsonl / episodes.jsonl / episodes_stats.jsonl

State features（dual-arm mobile robot）：
    observation.state.joint       float32[14]  左右臂各 7 关节角（rad）
    observation.gripper_position  float32[2]   左右夹爪总张开量（m）
    observation.state.end         float32[12]  左右 TCP xyz(m) + euler_xyz(rad)
    actions                       float32[16]  下一帧 joint(14) + gripper(2)

用法（从仓库根目录；macOS 离屏渲染使用 mjpython）：
    mjpython backend/serialize_to_lerobot.py
    mjpython backend/serialize_to_lerobot.py --src exports/myrun --fps 10
    mjpython backend/serialize_to_lerobot.py --only-passed
    mjpython backend/serialize_to_lerobot.py --overwrite
"""

import argparse
import datetime
import json
import math
import os
import sys

import mujoco
import numpy as np

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)

try:
    from .camera_config import CAMERA_CONFIG, RENDER_W, RENDER_H, load_scene_with_cameras
    from .robot_adapter import load_default_adapter
except ImportError:  # Direct script execution.
    from camera_config import CAMERA_CONFIG, RENDER_W, RENDER_H, load_scene_with_cameras
    from robot_adapter import load_default_adapter

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required: conda install pandas pyarrow")
try:
    from scipy.spatial.transform import Rotation
except ImportError:
    raise ImportError("scipy is required: conda install scipy")

# ── 常量 ────────────────────────────────────────────────────────────────────

CAMERAS             = list(CAMERA_CONFIG.keys())   # ["top", "left_wrist", "right_wrist"]
DEFAULT_FPS         = 10.0
CHUNKS_SIZE         = 1000
TASK_NAME           = "pick and place objects"


# ── 关节/site 索引 ───────────────────────────────────────────────────────────

def _build_jids(model, adapter):
    """提取左右臂关节 qpos 地址、夹爪 qpos 地址、TCP site id。"""
    def jqpos(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(model.jnt_qposadr[jid]) if jid >= 0 else None

    def sid(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        return int(i) if i >= 0 else None

    return {
        "left_arm": [jqpos(name) for name in adapter.get_arm_joints("left")],
        "right_arm": [jqpos(name) for name in adapter.get_arm_joints("right")],
        "left_grip": jqpos(adapter.get_gripper_joints("left")[0]),
        "right_grip": jqpos(adapter.get_gripper_joints("right")[0]),
        "left_grip_range": adapter.get_gripper_range("left"),
        "right_grip_range": adapter.get_gripper_range("right"),
        "left_tcp": sid(adapter.get_tcp_site("left")),
        "right_tcp": sid(adapter.get_tcp_site("right")),
    }


# ── 单帧特征提取 ─────────────────────────────────────────────────────────────

def _grip_to_m(position, configured_range):
    """Convert one finger position to total parallel-jaw opening in metres."""
    opened, closed = configured_range
    lo, hi = sorted((float(opened), float(closed)))
    return 2.0 * float(np.clip(position, lo, hi))


def _tcp_pose(model, data, site_id):
    """返回 TCP 世界坐标 xyz(m) + euler_xyz(rad) 共 6 维 float32。"""
    pos   = data.site_xpos[site_id].astype(np.float32)
    mat   = data.site_xmat[site_id].reshape(3, 3)
    euler = Rotation.from_matrix(mat).as_euler("xyz").astype(np.float32)
    return np.concatenate([pos, euler])


def _extract_frame(model, data, jids):
    """从当前 data 提取单帧状态，返回 dict(joint, gripper, end)。"""
    la = np.array([data.qpos[a] for a in jids["left_arm"]  if a is not None], dtype=np.float32)
    ra = np.array([data.qpos[a] for a in jids["right_arm"] if a is not None], dtype=np.float32)
    la = np.pad(la, (0, 7 - len(la))).astype(np.float32)
    ra = np.pad(ra, (0, 7 - len(ra))).astype(np.float32)

    lg = (_grip_to_m(data.qpos[jids["left_grip"]], jids["left_grip_range"])
          if jids["left_grip"] is not None else 0.0)
    rg = (_grip_to_m(data.qpos[jids["right_grip"]], jids["right_grip_range"])
          if jids["right_grip"] is not None else 0.0)

    lt = _tcp_pose(model, data, jids["left_tcp"])  if jids["left_tcp"]  is not None else np.zeros(6, np.float32)
    rt = _tcp_pose(model, data, jids["right_tcp"]) if jids["right_tcp"] is not None else np.zeros(6, np.float32)

    return {
        "joint":   np.concatenate([la, ra]),            # (14,)
        "gripper": np.array([lg, rg], dtype=np.float32), # (2,)
        "end":     np.concatenate([lt, rt]),             # (12,)
    }


# ── Replay：从 qpos 序列渲染每帧 ────────────────────────────────────────────

def _replay(model, data, jids, renderer, qpos_traj):
    """逐帧 set qpos → mj_forward → 渲染 + 提取状态。
    返回 (cam_frames {cam: list[H,W,3 uint8]}, states list[dict])。
    """
    cam_frames = {cam: [] for cam in CAMERAS}
    states = []
    for qpos in qpos_traj:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for cam in CAMERAS:
            renderer.update_scene(data, camera=cam)
            cam_frames[cam].append(renderer.render().copy())
        states.append(_extract_frame(model, data, jids))
    return cam_frames, states


# ── Video encoding via imageio-ffmpeg ────────────────────────────────────────

def _write_mp4(frames, out_path, fps):
    """frames: list[np.uint8 (H,W,3) RGB]，编码为 h264 MP4。"""
    import imageio
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec="libx264",
                            pixelformat="yuv420p",
                            output_params=["-crf", "18", "-preset", "fast"]) as w:
        for frame in frames:
            w.append_data(frame)


# ── Parquet 构建 ─────────────────────────────────────────────────────────────

def _build_df(states, ep_idx, global_frame_offset, fps):
    """从 per-frame states 构建 DataFrame。
    actions[t] = state[t+1] 的 joint+gripper（标准行为克隆；末帧用 state[T-1]）。
    """
    T = len(states)
    rows = []
    for t, s in enumerate(states):
        nxt    = states[min(t + 1, T - 1)]
        action = np.concatenate([nxt["joint"], nxt["gripper"]])  # (16,)
        rows.append({
            "timestamp":                    float(np.float32(t / fps)),
            "frame_index":                  int(t),
            "episode_index":                int(ep_idx),
            "index":                        int(global_frame_offset + t),
            "task_index":                   0,
            "observation.state.joint":      s["joint"].tolist(),
            "observation.gripper_position": s["gripper"].tolist(),
            "observation.state.end":        s["end"].tolist(),
            "actions":                      action.tolist(),
        })
    return pd.DataFrame(rows)


# ── Per-episode 统计 ─────────────────────────────────────────────────────────

def _compute_stats(df):
    stats = {}
    for col in ("observation.state.joint", "observation.gripper_position",
                "observation.state.end", "actions"):
        arr = np.array(df[col].tolist(), dtype=np.float32)
        stats[col] = {
            "mean":  arr.mean(axis=0).tolist(),
            "std":   arr.std(axis=0).tolist(),
            "min":   arr.min(axis=0).tolist(),
            "max":   arr.max(axis=0).tolist(),
            "count": [len(arr)],
        }
    return stats


# ── 单 episode 处理 ──────────────────────────────────────────────────────────

def _process_episode(ep_dir, ep_idx_out, global_frame_offset,
                     out_root, fps, model, data, renderer, jids):
    """处理一个 episode，写 parquet + 三路 MP4。
    返回 (n_frames, stats) 或 None（跳过）。
    """
    ann_path  = os.path.join(ep_dir, "annotation.json")
    traj_path = os.path.join(ep_dir, "trajectory_qpos.npy")

    if not os.path.isfile(ann_path):
        print("SKIP (no annotation.json)")
        return None

    # ── 获取 qpos 轨迹 ─────────────────────────────────────────────────────
    if not os.path.isfile(traj_path):
        print("SKIP (no trajectory_qpos.npy)")
        return None
    qpos_traj = np.load(traj_path)  # (T, nq)
    if qpos_traj.ndim != 2 or qpos_traj.shape[1] != model.nq:
        print(f"SKIP (trajectory shape {qpos_traj.shape}, expected (T, {model.nq}))")
        return None
    meta_path = os.path.join(ep_dir, "trajectory_meta.json")
    traj_fps  = fps
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        traj_fps = float(meta.get("record_fps", fps))

    if len(qpos_traj) == 0:
        print("SKIP (empty trajectory)")
        return None

    # ── Replay 渲染 ────────────────────────────────────────────────────────
    cam_frames, states = _replay(model, data, jids, renderer, qpos_traj)

    # ── 写 parquet ─────────────────────────────────────────────────────────
    chunk_idx   = ep_idx_out // CHUNKS_SIZE
    ep_tag      = f"episode_{ep_idx_out:06d}"
    parquet_dir = os.path.join(out_root, "data", f"chunk-{chunk_idx:03d}")
    os.makedirs(parquet_dir, exist_ok=True)

    df = _build_df(states, ep_idx_out, global_frame_offset, traj_fps)
    df.to_parquet(os.path.join(parquet_dir, f"{ep_tag}.parquet"), index=False)

    # ── 写 MP4 ─────────────────────────────────────────────────────────────
    for cam in CAMERAS:
        vid_path = os.path.join(out_root, "videos", f"chunk-{chunk_idx:03d}",
                                f"observation.image.{cam}", f"{ep_tag}.mp4")
        _write_mp4(cam_frames[cam], vid_path, traj_fps)

    return len(states), _compute_stats(df)


# ── Meta 文件写入 ────────────────────────────────────────────────────────────

def _write_meta(out_root, all_episodes_info, fps, adapter):
    """写全量 meta（info.json / tasks.jsonl / episodes.jsonl / episodes_stats.jsonl）。
    all_episodes_info: list of (ep_idx, n_frames, stats)，按 ep_idx 升序。
    """
    meta_dir = os.path.join(out_root, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    total_frames   = sum(n for _, n, _ in all_episodes_info)
    total_episodes = len(all_episodes_info)
    n_chunks       = max(1, math.ceil(total_episodes / CHUNKS_SIZE))

    left_joint_names = adapter.get_arm_joints("left")
    right_joint_names = adapter.get_arm_joints("right")
    joint_names = left_joint_names + right_joint_names
    action_names = joint_names + ["left_gripper_m", "right_gripper_m"]
    end_names = ["left_tcp_x",  "left_tcp_y",  "left_tcp_z",
                 "left_tcp_rx", "left_tcp_ry", "left_tcp_rz",
                 "right_tcp_x", "right_tcp_y", "right_tcp_z",
                 "right_tcp_rx","right_tcp_ry","right_tcp_rz"]

    features = {
        "observation.state.joint": {
            "dtype": "float32", "shape": [14], "names": joint_names},
        "observation.gripper_position": {
            "dtype": "float32", "shape": [2], "names": ["left_gripper_m", "right_gripper_m"]},
        "observation.state.end": {
            "dtype": "float32", "shape": [12], "names": end_names},
        "actions": {
            "dtype": "float32", "shape": [16], "names": action_names},
        "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
        "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
        "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
        "index":         {"dtype": "int64",   "shape": [1], "names": None},
        "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
    }
    for cam in CAMERAS:
        features[f"observation.image.{cam}"] = {
            "dtype": "video",
            "shape": [RENDER_H, RENDER_W, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": RENDER_H, "video.width": RENDER_W, "video.channels": 3,
                "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.fps": float(fps), "video.is_depth_map": False,
            },
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": adapter.robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(CAMERAS),
        "total_chunks": n_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": float(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(os.path.join(meta_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_NAME}) + "\n")

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for ep_idx, n_frames, _ in all_episodes_info:
            f.write(json.dumps({"episode_index": ep_idx,
                                "tasks": [TASK_NAME], "length": n_frames}) + "\n")

    with open(os.path.join(meta_dir, "episodes_stats.jsonl"), "w", encoding="utf-8") as f:
        for ep_idx, _, stats in all_episodes_info:
            f.write(json.dumps({"episode_index": ep_idx, "stats": stats}) + "\n")


# ── Append 模式：加载已有 meta ───────────────────────────────────────────────

def _load_existing_meta(out_root):
    """读取已有的 episodes.jsonl + episodes_stats.jsonl，返回 (list, int, int)：
    existing_info list[(ep_idx, n_frames, stats)], next_ep_idx, total_frames_so_far。
    """
    ep_path = os.path.join(out_root, "meta", "episodes.jsonl")
    st_path = os.path.join(out_root, "meta", "episodes_stats.jsonl")
    if not os.path.isfile(ep_path):
        return [], 0, 0

    ep_lines = [json.loads(l) for l in open(ep_path) if l.strip()]
    st_lines = [json.loads(l) for l in open(st_path) if l.strip()] if os.path.isfile(st_path) else []
    st_by_idx = {s["episode_index"]: s["stats"] for s in st_lines}

    existing = []
    total_frames = 0
    for ep in ep_lines:
        idx = ep["episode_index"]
        nf  = ep["length"]
        stats = st_by_idx.get(idx, {})
        existing.append((idx, nf, stats))
        total_frames += nf

    next_idx = max((e[0] for e in existing), default=-1) + 1
    return existing, next_idx, total_frames


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert exports/ to LeRobot v2.1 dataset.")
    parser.add_argument("--src",         default=os.path.join(REPO_ROOT, "exports"),
                        help="包含 episode_XXXXXX/qpos 轨迹的输入目录")
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out",         default=os.path.join(REPO_ROOT, "data", f"lerobot_{_ts}"),
                        help="LeRobot 输出目录（默认按时间戳自动命名 data/lerobot_YYYYMMDD_HHMMSS/）")
    parser.add_argument("--fps",         type=float, default=DEFAULT_FPS,
                        help="目标帧率 Hz（默认 10）")
    parser.add_argument("--only-passed", action="store_true",
                        help="只处理 success=true 的 episode")
    parser.add_argument("--overwrite",   action="store_true",
                        help="从 ep_idx=0 重新写（默认 append）")
    parser.add_argument("--upload-oss",  action="store_true",
                        help="转换完成后上传到 OSS（需设 OSS_* 环境变量）")
    parser.add_argument("--oss-prefix",  default="",
                        help="OSS 对象前缀/目录，例如 lerobot/dual_arm_robot/（覆盖 OSS_PREFIX 环境变量）")
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        print(f"[ERR] src dir not found: {args.src}")
        return

    src_eps = sorted([
        d for d in os.listdir(args.src)
        if d.startswith("episode_") and os.path.isdir(os.path.join(args.src, d))
    ])
    if not src_eps:
        print(f"[ERR] no episode_XXXXXX dirs in {args.src}")
        return

    # 加载已有 meta（append 模式）
    if args.overwrite:
        existing_info, ep_idx_start, frame_offset = [], 0, 0
    else:
        # schema 兼容检查：如果已有 info.json 的 joint shape 与新 schema 不同，终止
        info_path = os.path.join(args.out, "meta", "info.json")
        if os.path.isfile(info_path):
            with open(info_path) as f:
                old_info = json.load(f)
            old_joint_shape = (old_info.get("features", {})
                               .get("observation.state.joint", {})
                               .get("shape", []))
            if old_joint_shape and old_joint_shape != [14]:
                print(f"[ERR] schema mismatch: existing dataset has joint shape "
                      f"{old_joint_shape}, this serializer outputs [14].")
                print("      Use --overwrite to start a fresh dataset.")
                return
        existing_info, ep_idx_start, frame_offset = _load_existing_meta(args.out)

    print(f"src : {args.src}  ({len(src_eps)} episodes)")
    print(f"out : {args.out}  (append from ep_idx={ep_idx_start})")
    print(f"fps : {args.fps}  |  only-passed: {args.only_passed}")

    # 建共用 model/renderer（避免每 episode 重建）
    adapter     = load_default_adapter()
    model, data = load_scene_with_cameras(CAMERA_CONFIG)
    jids        = _build_jids(model, adapter)
    renderer    = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    new_episodes_info = []
    global_frame_offset = frame_offset

    for ep_dir_name in src_eps:
        ep_dir    = os.path.join(args.src, ep_dir_name)
        ep_idx_out = ep_idx_start + len(new_episodes_info)

        # --only-passed 过滤
        ann_path = os.path.join(ep_dir, "annotation.json")
        if args.only_passed and os.path.isfile(ann_path):
            with open(ann_path) as f:
                check = json.load(f)
            if not check.get("success", False):
                print(f"  skip {ep_dir_name} (success=False)")
                continue

        print(f"  [{ep_idx_out:04d}] {ep_dir_name} ...", end=" ", flush=True)
        result = _process_episode(
            ep_dir, ep_idx_out, global_frame_offset,
            args.out, args.fps,
            model, data, renderer, jids,
        )
        if result is None:
            continue
        n_frames, stats = result
        global_frame_offset += n_frames
        new_episodes_info.append((ep_idx_out, n_frames, stats))
        print(f"OK  ({n_frames} frames)")

    renderer.close()

    if not new_episodes_info:
        print("[WARN] no episodes processed")
        return

    all_info = existing_info + new_episodes_info
    _write_meta(args.out, all_info, args.fps, adapter)

    print(f"\n{'='*40}")
    print(f"新增 {len(new_episodes_info)} episodes → {args.out}")
    print(f"数据集共 {len(all_info)} episodes，{sum(n for _,n,_ in all_info)} frames")

    if args.upload_oss:
        _upload_oss(args.out, args.oss_prefix)


def _upload_oss(local_dir: str, prefix_override: str = ""):
    """将 local_dir 拷贝到 DSW 挂载的 OSS 目录。

    DSW 已通过 ossfs2 将 OSS bucket 挂载在 /mnt/data，直接文件拷贝即可，
    无需 oss2 SDK。目标路径：/mnt/data/output/lerobot/<dataset_name>/
    可通过 --oss-prefix 或 OSS_MOUNT_PREFIX 环境变量覆盖子目录前缀。
    """
    import shutil

    mount_root = os.environ.get("OSS_MOUNT_ROOT", "/mnt/data/output")
    prefix     = prefix_override or os.environ.get("OSS_PREFIX", "lerobot")
    local_root = os.path.abspath(local_dir)
    dataset_name = os.path.basename(local_root)
    dst_root = os.path.join(mount_root, prefix, dataset_name)

    if not os.path.isdir(mount_root):
        print(f"[OSS] 挂载点不存在：{mount_root}，跳过上传")
        return

    print(f"\n[OSS] 同步 → {dst_root}")
    files = []
    for dirpath, _, filenames in os.walk(local_root):
        for fname in filenames:
            files.append(os.path.join(dirpath, fname))

    ok = fail = 0
    for fpath in files:
        rel = os.path.relpath(fpath, local_root)
        dst = os.path.join(dst_root, rel)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(fpath, dst)
            ok += 1
            if ok % 50 == 0:
                print(f"  {ok}/{len(files)} 已拷贝…", flush=True)
        except Exception as e:
            print(f"  [WARN] 拷贝失败 {rel}: {e}")
            fail += 1

    print(f"[OSS] 完成：{ok} 成功，{fail} 失败 → {dst_root}")


if __name__ == "__main__":
    main()
