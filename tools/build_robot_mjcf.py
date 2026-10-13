#!/usr/bin/env python3
"""
Build MuJoCo MJCF from URDF with actuators, grippers, and scene elements

This script is the ONLY way to generate robot/dual_arm_robot.xml
Never hand-edit the output XML - it will be overwritten on next build.

Usage:
    python tools/build_robot_mjcf.py
    python tools/build_robot_mjcf.py --config robot/robot_config.yaml
"""

import mujoco
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
from robot_adapter import RobotAdapter


# ============ Configuration ============
URDF_PATH = 'robot/dual_arm_robot.urdf'
OUTPUT_PATH = 'robot/dual_arm_robot.xml'
CONFIG_PATH = 'robot/robot_config.yaml'

# Scene parameters
BASE_INIT_Z = 0.5  # Initial height above floor
FLOOR_SIZE = [10, 10, 0.1]

# Gripper parameters (simple parallel gripper)
GRIPPER_WIDTH = 0.04  # Width when open
GRIPPER_FINGER_HALF_SIZE = [0.006, 0.008, 0.08]
TCP_OFFSET = [0.0, 0.0, 0.26]
GRIPPER_BASE_Z = 0.10
GRIPPER_KP = 50.0
GRIPPER_KV = 5.0


def load_config(config_path: str = CONFIG_PATH):
    """Load robot configuration"""
    try:
        adapter = RobotAdapter(config_path)
        print(f"Loaded config: {adapter}")
        return adapter
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        print("Using default parameters")
        return None


def urdf_to_mjcf_xml(urdf_path: str) -> ET.Element:
    """
    Convert URDF to MJCF XML string using MuJoCo's built-in converter

    Returns:
        XML Element tree
    """
    print(f"Converting URDF: {urdf_path}")

    # Use MjSpec to convert URDF
    spec = mujoco.MjSpec()
    spec.from_file(urdf_path)

    # Get XML string
    xml_string = spec.to_xml()

    # Parse to ElementTree
    root = ET.fromstring(xml_string)

    print(f"  -> Converted to MJCF ({len(xml_string)} bytes)")
    return root


def add_compiler_options(root: ET.Element):
    """Add compiler options for better mesh handling"""
    compiler = root.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(root, 'compiler', attrib={
            'angle': 'radian',
            'meshdir': 'meshes',
            'eulerseq': 'xyz'
        })
    else:
        compiler.set('angle', 'radian')
        compiler.set('eulerseq', 'xyz')


def add_scene_elements(root: ET.Element, adapter: RobotAdapter = None):
    """Add floor, lights, and simulation options"""

    # Option settings
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')

    option.set('timestep', '0.002')
    option.set('iterations', '50')
    option.set('solver', 'Newton')
    option.set('gravity', '0 0 -9.81')

    # Visual settings
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')

    # Worldbody modifications
    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root, 'worldbody')

    # Add floor
    floor = worldbody.find(".//geom[@name='floor']")
    if floor is None:
        floor = ET.SubElement(worldbody, 'geom', attrib={
            'name': 'floor',
            'type': 'plane',
            'size': f'{FLOOR_SIZE[0]} {FLOOR_SIZE[1]} {FLOOR_SIZE[2]}',
            'rgba': '0.9 0.9 0.9 1',
            'friction': '1.0 0.005 0.0001'
        })
        print("  Added floor")

    # Add lights
    light_positions = [
        ('light1', '2 2 3'),
        ('light2', '-2 2 3'),
        ('light3', '0 -2 3')
    ]

    for light_name, pos in light_positions:
        if worldbody.find(f".//light[@name='{light_name}']") is None:
            ET.SubElement(worldbody, 'light', attrib={
                'name': light_name,
                'pos': pos,
                'dir': '0 0 -1',
                'diffuse': '0.8 0.8 0.8',
                'specular': '0.2 0.2 0.2'
            })

    print("  Added scene elements (floor, lights)")

    # Set base initial height
    base_body = worldbody.find(".//body[@name='base_link']")
    if base_body is not None:
        # Add freejoint for mobile base
        if base_body.find('freejoint') is None:
            ET.SubElement(base_body, 'freejoint', attrib={'name': 'base_freejoint'})
            print("  Added base freejoint")

        # Set initial position
        current_pos = base_body.get('pos', '0 0 0').split()
        current_pos[2] = str(BASE_INIT_Z)
        base_body.set('pos', ' '.join(current_pos))
        print(f"  Set base initial height: {BASE_INIT_Z}m")


def compile_model_for_introspection(urdf_path: str) -> mujoco.MjModel:
    """
    Compile URDF to MjModel for joint introspection
    This is SEPARATE from the MjSpec flow to avoid corruption
    """
    return mujoco.MjModel.from_xml_path(urdf_path)


def add_actuators(root: ET.Element, model: mujoco.MjModel, adapter: RobotAdapter = None):
    """
    Add actuators for all joints based on compiled model

    Position servos for arms/torso, velocity servos for wheels
    """
    actuator_section = root.find('actuator')
    if actuator_section is None:
        actuator_section = ET.SubElement(root, 'actuator')

    # Get actuator params from config
    if adapter and 'actuators' in adapter.config:
        act_cfg = adapter.config['actuators']
        pos_kp = act_cfg.get('position_servos', {}).get('kp', 100.0)
        pos_kv = act_cfg.get('position_servos', {}).get('kv', 10.0)
        vel_kv = act_cfg.get('velocity_servos', {}).get('kv', 1.0)
        wheel_ctrlrange = act_cfg.get('velocity_servos', {}).get('ctrlrange', 10.0)
    else:
        pos_kp = 100.0
        pos_kv = 10.0
        vel_kv = 1.0
        wheel_ctrlrange = 10.0

    count_pos = 0
    count_vel = 0

    for jid in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)

        if not joint_name:
            continue

        # Skip freejoint
        if 'free' in joint_name.lower():
            continue

        joint_type = model.jnt_type[jid]

        # Velocity servo for wheels (continuous joints)
        if 'wheel' in joint_name.lower() and joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            actuator_name = f"vel_{joint_name}"

            if actuator_section.find(f".//velocity[@name='{actuator_name}']") is None:
                ET.SubElement(actuator_section, 'velocity', attrib={
                    'name': actuator_name,
                    'joint': joint_name,
                    'kv': str(vel_kv),
                    'ctrlrange': f'-{wheel_ctrlrange} {wheel_ctrlrange}'
                })
                count_vel += 1

        # Position servo for everything else
        elif joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            actuator_name = f"pos_{joint_name}"

            if actuator_section.find(f".//position[@name='{actuator_name}']") is None:
                # Get joint limits from model
                jnt_limited = model.jnt_limited[jid]
                if jnt_limited:
                    jnt_range = model.jnt_range[jid]
                    ctrlrange = f"{jnt_range[0]} {jnt_range[1]}"
                else:
                    ctrlrange = "-3.14 3.14"  # Default for unlimited

                ET.SubElement(actuator_section, 'position', attrib={
                    'name': actuator_name,
                    'joint': joint_name,
                    'kp': str(pos_kp),
                    'kv': str(pos_kv),
                    'ctrlrange': ctrlrange
                })
                count_pos += 1

    print(f"  Added {count_pos} position servos, {count_vel} velocity servos")


def add_tcp_sites(root: ET.Element, adapter: RobotAdapter = None):
    """Add TCP (Tool Center Point) sites to wrist links"""
    worldbody = root.find('worldbody')

    tcp_offset = (
        adapter.tool_geometry.get('tcp_offset', TCP_OFFSET)
        if adapter else TCP_OFFSET)

    for side in ['left', 'right']:
        wrist_link = f"{side}_link7"
        tcp_name = f"{side}_tcp"

        # Find wrist body
        wrist_body = worldbody.find(f".//body[@name='{wrist_link}']")
        if wrist_body is not None:
            # Check if TCP site already exists
            if wrist_body.find(f".//site[@name='{tcp_name}']") is None:
                ET.SubElement(wrist_body, 'site', attrib={
                    'name': tcp_name,
                    'pos': f"{tcp_offset[0]} {tcp_offset[1]} {tcp_offset[2]}",
                    'type': 'sphere',
                    'size': '0.01',
                    'rgba': '1 0 0 0.5'
                })
                print(f"  Added TCP site: {tcp_name}")


def add_simple_grippers(root: ET.Element, adapter: RobotAdapter = None):
    """
    Add simple parallel grippers (two fingers) to both wrists

    This creates basic gripper geometry directly in the MJCF
    """
    worldbody = root.find('worldbody')
    actuator_section = root.find('actuator')
    if actuator_section is None:
        actuator_section = ET.SubElement(root, 'actuator')

    for side in ['left', 'right']:
        wrist_link = f"{side}_link7"
        wrist_body = worldbody.find(f".//body[@name='{wrist_link}']")

        if wrist_body is None:
            print(f"  Warning: {wrist_link} not found, skipping gripper")
            continue

        # Gripper base offset (from wrist)
        tool_geometry = adapter.tool_geometry if adapter else {}
        gripper_base_z = float(
            tool_geometry.get('finger_base_z', GRIPPER_BASE_Z))
        finger_half_size = tool_geometry.get(
            'finger_half_size', GRIPPER_FINGER_HALF_SIZE)

        # Left finger
        left_finger_name = f"{side}_gripper_left_finger"
        left_finger_joint = f"{side}_gripper_left_finger_joint"

        if wrist_body.find(f".//body[@name='{left_finger_name}']") is None:
            left_finger = ET.SubElement(wrist_body, 'body', attrib={
                'name': left_finger_name,
                'pos': f"{-GRIPPER_WIDTH/2} 0 {gripper_base_z}"
            })

            # Joint (prismatic, moves outward)
            ET.SubElement(left_finger, 'joint', attrib={
                'name': left_finger_joint,
                'type': 'slide',
                'axis': '-1 0 0',
                'range': f'0 {GRIPPER_WIDTH/2}',
                'damping': '10.0'
            })

            # Finger geometry
            ET.SubElement(left_finger, 'geom', attrib={
                'type': 'box',
                'size': ' '.join(str(value) for value in finger_half_size),
                'pos': f'0 0 {finger_half_size[2]}',
                'rgba': '0.3 0.3 0.3 1',
                'friction': '2.0 0.005 0.0001'
            })

        # Right finger (mirrored)
        right_finger_name = f"{side}_gripper_right_finger"
        right_finger_joint = f"{side}_gripper_right_finger_joint"

        if wrist_body.find(f".//body[@name='{right_finger_name}']") is None:
            right_finger = ET.SubElement(wrist_body, 'body', attrib={
                'name': right_finger_name,
                'pos': f"{GRIPPER_WIDTH/2} 0 {gripper_base_z}"
            })

            ET.SubElement(right_finger, 'joint', attrib={
                'name': right_finger_joint,
                'type': 'slide',
                'axis': '1 0 0',
                'range': f'0 {GRIPPER_WIDTH/2}',
                'damping': '10.0'
            })

            ET.SubElement(right_finger, 'geom', attrib={
                'type': 'box',
                'size': ' '.join(str(value) for value in finger_half_size),
                'pos': f'0 0 {finger_half_size[2]}',
                'rgba': '0.3 0.3 0.3 1',
                'friction': '2.0 0.005 0.0001'
            })

        # Add gripper actuators (both fingers move together)
        gripper_actuator_name = f"{side}_gripper"

        # Position servo controlling both fingers via equality constraint
        if actuator_section.find(f".//position[@name='{gripper_actuator_name}']") is None:
            ET.SubElement(actuator_section, 'position', attrib={
                'name': gripper_actuator_name,
                'joint': left_finger_joint,  # Control left finger directly
                'kp': str(GRIPPER_KP),
                'kv': str(GRIPPER_KV),
                'ctrlrange': f'0 {GRIPPER_WIDTH/2}'
            })

        # Add equality constraint to couple right finger to left
        equality_section = root.find('equality')
        if equality_section is None:
            equality_section = ET.SubElement(root, 'equality')

        eq_name = f"{side}_gripper_couple"
        if equality_section.find(f".//joint[@name='{eq_name}']") is None:
            ET.SubElement(equality_section, 'joint', attrib={
                'name': eq_name,
                'joint1': left_finger_joint,
                'joint2': right_finger_joint,
                'polycoef': '0 1 0 0 0'  # q2 = q1
            })

    print("  Added simple parallel grippers")


def hide_collision_geoms(root: ET.Element):
    """Make collision geometries invisible (set alpha=0)"""
    worldbody = root.find('worldbody')
    count = 0

    for geom in worldbody.iter('geom'):
        # Skip floor
        if geom.get('name') == 'floor':
            continue

        # Set collision geoms to invisible (keep visual mesh visible)
        geom_type = geom.get('type', 'box')
        if geom_type in ['box', 'cylinder', 'capsule', 'sphere']:
            rgba = geom.get('rgba', '1 1 1 1').split()
            rgba[3] = '0'  # Set alpha to 0
            geom.set('rgba', ' '.join(rgba))
            count += 1

    print(f"  Made {count} collision geoms invisible")


def add_contact_excludes(root: ET.Element, adapter: RobotAdapter = None):
    """Add contact exclusions to prevent self-collision artifacts"""
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')

    # Default exclude pairs (adjacent links)
    default_excludes = [
        ('left_link2', 'left_link3'),
        ('left_link3', 'left_link4'),
        ('left_link4', 'left_link5'),
        ('left_link5', 'left_link6'),
        ('left_link6', 'left_link7'),
        ('right_link2', 'right_link3'),
        ('right_link3', 'right_link4'),
        ('right_link4', 'right_link5'),
        ('right_link5', 'right_link6'),
        ('right_link6', 'right_link7'),
        ('torso_link', 'left_link1'),
        ('torso_link', 'right_link1'),
    ]

    # Get additional excludes from config
    if adapter and 'collision' in adapter.config:
        exclude_pairs = adapter.config['collision'].get('exclude_pairs', [])
    else:
        exclude_pairs = default_excludes

    count = 0
    for body1, body2 in exclude_pairs:
        exclude_name = f"exclude_{body1}_{body2}"
        if contact.find(f".//exclude[@name='{exclude_name}']") is None:
            ET.SubElement(contact, 'exclude', attrib={
                'name': exclude_name,
                'body1': body1,
                'body2': body2
            })
            count += 1

    print(f"  Added {count} contact exclusions")


def prettify_xml(elem: ET.Element, level=0):
    """Add indentation to XML for readability"""
    indent = "  "
    i = "\n" + level * indent
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            prettify_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def main():
    parser = argparse.ArgumentParser(description='Build MuJoCo MJCF from URDF')
    parser.add_argument('--config', default=CONFIG_PATH, help='Robot config YAML')
    parser.add_argument('--urdf', default=URDF_PATH, help='Input URDF path')
    parser.add_argument('--output', default=OUTPUT_PATH, help='Output MJCF path')
    args = parser.parse_args()

    print("=" * 60)
    print("Building MuJoCo MJCF from URDF")
    print("=" * 60)

    # Load config
    adapter = load_config(args.config)

    # Step 1: Convert URDF to MJCF XML
    print("\n[1/9] Converting URDF to MJCF...")
    root = urdf_to_mjcf_xml(args.urdf)

    # Step 2: Compile model for introspection (separate from MjSpec)
    print("\n[2/9] Compiling model for joint introspection...")
    model = compile_model_for_introspection(args.urdf)
    print(f"  Model has {model.njnt} joints, {model.nbody} bodies")

    # Step 3: Add compiler options
    print("\n[3/9] Adding compiler options...")
    add_compiler_options(root)

    # Step 4: Add scene elements
    print("\n[4/9] Adding scene elements...")
    add_scene_elements(root, adapter)

    # Step 5: Add actuators
    print("\n[5/9] Adding actuators...")
    add_actuators(root, model, adapter)

    # Step 6: Add TCP sites
    print("\n[6/9] Adding TCP sites...")
    add_tcp_sites(root, adapter)

    # Step 7: Add grippers
    print("\n[7/9] Adding grippers...")
    add_simple_grippers(root, adapter)

    # Step 8: Hide collision geoms
    print("\n[8/9] Hiding collision geometries...")
    hide_collision_geoms(root)

    # Step 9: Add contact exclusions
    print("\n[9/9] Adding contact exclusions...")
    add_contact_excludes(root, adapter)

    # Prettify and save
    print(f"\nWriting MJCF to: {args.output}")
    prettify_xml(root)
    tree = ET.ElementTree(root)
    tree.write(args.output, encoding='utf-8', xml_declaration=True)

    # Verify output can be loaded
    print("\nVerifying output...")
    try:
        verify_model = mujoco.MjModel.from_xml_path(args.output)
        print(f"  Success! Model has:")
        print(f"    - {verify_model.njnt} joints")
        print(f"    - {verify_model.nu} actuators")
        print(f"    - {verify_model.nbody} bodies")
        print(f"    - {verify_model.ngeom} geometries")
    except Exception as e:
        print(f"  ERROR: Could not load generated MJCF: {e}")
        return 1

    print("\n" + "=" * 60)
    print("Build complete!")
    print("=" * 60)
    print(f"\nTo view: python -m mujoco.viewer --mjcf={args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
