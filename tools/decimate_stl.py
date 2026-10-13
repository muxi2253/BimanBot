"""把面片数超过 MuJoCo 20万上限的二进制 STL 精简到目标面数以内。

用体素聚类(voxel clustering):把顶点吸附到一个网格上合并,丢掉合并后退化
(两个角重合)的三角形,再去重。不追求最佳保真度,只求快速、无额外依赖
地把面数砍下来——CAD 直出的网格通常有大量远超渲染/碰撞需要的细节,能砍
掉一大截也不会看出来。

用法:
    python tools/decimate_stl.py path/to/mesh.STL
    python tools/decimate_stl.py path/to/mesh.STL -o out.STL -t 100000

不传 -o 时默认写到 <文件名>_simplified.STL,不覆盖原文件。
"""
import argparse
import struct
import sys

import numpy as np

MUJOCO_FACE_LIMIT = 200_000
DEFAULT_TARGET_FACES = 180_000  # 留一点余量,不要正好卡在上限


def read_binary_stl(path):
    with open(path, 'rb') as f:
        header = f.read(80)
        nfaces = struct.unpack('<I', f.read(4))[0]
        data = f.read(nfaces * 50)
    arr = np.frombuffer(data, dtype=np.uint8).reshape(nfaces, 50)
    floats = np.frombuffer(arr[:, 0:48].tobytes(), dtype='<f4').reshape(nfaces, 4, 3)
    tris = floats[:, 1:4, :]  # (nfaces, 3 verts, 3 xyz),法线不读,反正下面会重新算
    return header, tris


def write_binary_stl(path, header, tris):
    nfaces = tris.shape[0]
    v0 = tris[:, 1, :] - tris[:, 0, :]
    v1 = tris[:, 2, :] - tris[:, 0, :]
    normals = np.cross(v0, v1)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    normals = (normals / norm_len).astype('<f4')
    with open(path, 'wb') as f:
        f.write(header)
        f.write(struct.pack('<I', nfaces))
        for i in range(nfaces):
            f.write(normals[i].tobytes())
            f.write(tris[i, 0].astype('<f4').tobytes())
            f.write(tris[i, 1].astype('<f4').tobytes())
            f.write(tris[i, 2].astype('<f4').tobytes())
            f.write(struct.pack('<H', 0))


def _decimate_once(tris, voxel):
    flat = tris.reshape(-1, 3)
    snapped = np.round(flat / voxel).astype(np.int64) * voxel
    snapped = snapped.astype('<f4').reshape(tris.shape)
    a, b, c = snapped[:, 0], snapped[:, 1], snapped[:, 2]
    degenerate = np.all(a == b, axis=1) | np.all(b == c, axis=1) | np.all(a == c, axis=1)
    kept = snapped[~degenerate]
    # 吸附之后可能出现完全重复的三角形,顺手去重
    key = np.sort(kept.reshape(kept.shape[0], -1), axis=1)
    _, unique_idx = np.unique(key, axis=0, return_index=True)
    return kept[np.sort(unique_idx)]


def simplify_stl(path, out_path=None, target_faces=DEFAULT_TARGET_FACES, max_attempts=20):
    """把 path 精简到 target_faces 面以内,写到 out_path(默认 <stem>_simplified.STL)。

    输入已经在目标面数以内时直接跳过,返回 None。
    """
    header, tris = read_binary_stl(path)
    n_in = tris.shape[0]
    if n_in <= target_faces:
        print(f'{path}: {n_in} faces, already under {target_faces}, skipping')
        return None

    if out_path is None:
        if path.lower().endswith('.stl'):
            out_path = path[:-4] + '_simplified' + path[-4:]
        else:
            out_path = path + '_simplified'

    diag = np.linalg.norm(tris.reshape(-1, 3).max(axis=0) - tris.reshape(-1, 3).min(axis=0))
    print(f'input faces: {n_in}, bbox diag: {diag:.4f} m')

    voxel = diag / 800.0
    result = tris
    for _ in range(max_attempts):
        result = _decimate_once(tris, voxel)
        print(f'  voxel={voxel:.6f} -> faces={result.shape[0]}')
        if result.shape[0] <= target_faces:
            break
        voxel *= 1.35
    else:
        print('warning: did not converge under target face count, using last attempt', file=sys.stderr)

    if result.shape[0] > MUJOCO_FACE_LIMIT:
        print(f'warning: simplified mesh still has {result.shape[0]} faces, over MuJoCo\'s '
              f'{MUJOCO_FACE_LIMIT} limit - try a larger voxel or lower -t', file=sys.stderr)

    write_binary_stl(out_path, header, result)
    print(f'wrote {out_path}: {result.shape[0]} faces')
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='输入的二进制 STL 文件')
    parser.add_argument('-o', '--output', default=None, help='输出路径,默认 <文件名>_simplified.STL')
    parser.add_argument('-t', '--target', type=int, default=DEFAULT_TARGET_FACES,
                         help=f'目标面数上限,默认 {DEFAULT_TARGET_FACES}')
    args = parser.parse_args()
    simplify_stl(args.input, args.output, args.target)


if __name__ == '__main__':
    main()
