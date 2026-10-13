"""
从视觉 mesh 的编译后顶点推导 proxy 碰撞体的 size / pos / quat（body 坐标系下的 MJCF 格式）。

用法：
    python tools/derive_proxy_from_mesh.py <body_name>
    python tools/derive_proxy_from_mesh.py <body_name> --type cylinder
    python tools/derive_proxy_from_mesh.py <body_name> --scene path/to/scene.xml --type cylinder

示例：
    python tools/derive_proxy_from_mesh.py chouzhi_0
    python tools/derive_proxy_from_mesh.py kebike_0 --type cylinder
"""
import sys
import argparse
import pathlib
import numpy as np
import mujoco

ROOT = pathlib.Path(__file__).parent.parent
DEFAULT_XML = ROOT / "scene" / "combined_scene.xml"

# cylinder 默认长轴为 Z；将 Z 轴转到其他轴所需的 quat（MJCF w-first，未归一化，MuJoCo 自动归一化）
_CYLINDER_QUAT = {
    "X": "0.70711 0 0.70711 0",   # 绕 Y 轴 90°，Z → X
    "Y": "0 0 1 1",               # 绕 (Y+Z)/√2 轴 180°，Z → Y
    "Z": None,                     # 无需旋转
}


def quat_to_mat(q):
    """MJCF w-first 四元数 → 3×3 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def derive_proxy(body_name: str, xml_path: str, proxy_type: str):
    m = mujoco.MjModel.from_xml_path(xml_path)

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        print(f"ERROR: body '{body_name}' not found")
        sys.exit(1)

    mesh_geoms = [
        g for g in range(m.ngeom)
        if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    if not mesh_geoms:
        print(f"ERROR: no mesh geom under body '{body_name}'")
        sys.exit(1)

    all_verts = []
    for gid in mesh_geoms:
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        mid = m.geom_dataid[gid]
        if mid < 0:
            continue
        start = m.mesh_vertadr[mid]
        count = m.mesh_vertnum[mid]
        local_v = m.mesh_vert[start : start + count].copy()

        gpos  = m.geom_pos[gid]
        gquat = m.geom_quat[gid]
        R = quat_to_mat(gquat)
        body_v = (R @ local_v.T).T + gpos
        all_verts.append(body_v)
        print(f"  geom '{gname}': {count} verts, geom_pos={gpos.round(4)}")

    verts = np.vstack(all_verts)
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center       = (vmin + vmax) / 2
    half_extents = (vmax - vmin) / 2

    print()
    print(f"=== proxy 推导结果（body='{body_name}'，type={proxy_type}）===")
    print(f"  bounding box min: {(vmin*1000).round(1)} mm")
    print(f"  bounding box max: {(vmax*1000).round(1)} mm")
    print(f"  full size : {(half_extents*2000).round(1)} mm  (X Y Z)")
    print()

    if proxy_type == "box":
        print(f'  MJCF:  type="box"')
        print(f'         size="{half_extents[0]:.5f} {half_extents[1]:.5f} {half_extents[2]:.5f}"')
        print(f'         pos="{center[0]:.5f} {center[1]:.5f} {center[2]:.5f}"')
    else:
        axis_idx  = int(np.argmax(half_extents))
        axis_name = "XYZ"[axis_idx]
        half_h    = half_extents[axis_idx]
        radius    = float(np.mean([half_extents[i] for i in range(3) if i != axis_idx]))
        quat      = _CYLINDER_QUAT[axis_name]

        print(f'  long axis: {axis_name}  ({half_h*2000:.1f} mm),  radius: {radius*1000:.1f} mm')
        print()
        print(f'  MJCF:  type="cylinder"')
        print(f'         size="{radius:.5f} {half_h:.5f}"')
        print(f'         pos="{center[0]:.5f} {center[1]:.5f} {center[2]:.5f}"')
        if quat:
            print(f'         quat="{quat}"')
        else:
            print(f'         (long axis is Z, no quat needed)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 mesh 推导 MJCF proxy 碰撞体参数")
    parser.add_argument("body", help="body 名称，如 kebike_0")
    parser.add_argument("--scene", "-s", default=str(DEFAULT_XML),
                        metavar="XML", help="场景 XML 路径（默认：scene/combined_scene.xml）")
    parser.add_argument("--type", "-t", choices=["box", "cylinder"], default="box",
                        dest="proxy_type", metavar="TYPE",
                        help="proxy 类型：box（默认）或 cylinder")
    args = parser.parse_args()

    print(f"Loading: {args.scene}\nBody:    {args.body}\n")
    derive_proxy(args.body, args.scene, args.proxy_type)
