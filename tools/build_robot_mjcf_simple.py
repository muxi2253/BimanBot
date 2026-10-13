#!/usr/bin/env python3
"""
Simplified MJCF builder - works WITHOUT mujoco installed
Directly manipulates XML to add actuators and scene elements

This version is for bootstrapping when mujoco is not yet installed.
For production use, run the full build_robot_mjcf.py after installing mujoco.

Usage:
    python tools/build_robot_mjcf_simple.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import sys

import yaml

# Configuration
URDF_PATH = 'robot/dual_arm_robot.urdf'
OUTPUT_PATH = 'robot/dual_arm_robot.xml'
BASE_INIT_Z = 0.5
FLOOR_SIZE = [10, 10, 0.1]
CONFIG_PATH = 'robot/robot_config.yaml'


def load_tool_geometry():
    """Load the tool dimensions without importing MuJoCo-dependent modules."""
    with open(CONFIG_PATH, encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    return config.get('tool_geometry', {})


def parse_urdf(urdf_path: str) -> ET.Element:
    """Parse URDF file"""
    tree = ET.parse(urdf_path)
    return tree.getroot()


def urdf_to_mjcf(urdf_root: ET.Element) -> ET.Element:
    """Convert URDF structure to MJCF structure"""
    # Create MJCF root
    mjcf = ET.Element('mujoco', attrib={'model': 'dual_arm_robot'})

    # Add compiler
    ET.SubElement(mjcf, 'compiler', attrib={
        'angle': 'radian',
        'eulerseq': 'xyz'
    })

    # Add option
    ET.SubElement(mjcf, 'option', attrib={
        'timestep': '0.002',
        'iterations': '50',
        'solver': 'Newton',
        'gravity': '0 0 -9.81'
    })

    # Add assets (materials)
    asset = ET.SubElement(mjcf, 'asset')
    materials = {
        'base_grey': '0.5 0.5 0.5 1',
        'wheel_black': '0.2 0.2 0.2 1',
        'torso_grey': '0.6 0.6 0.6 1',
        'arm_blue': '0.3 0.5 0.8 1',
        'gripper_grey': '0.7 0.7 0.7 1'
    }
    for name, rgba in materials.items():
        ET.SubElement(asset, 'material', attrib={'name': name, 'rgba': rgba})

    # Add worldbody
    worldbody = ET.SubElement(mjcf, 'worldbody')

    # Add floor
    ET.SubElement(worldbody, 'geom', attrib={
        'name': 'floor',
        'type': 'plane',
        'size': f'{FLOOR_SIZE[0]} {FLOOR_SIZE[1]} {FLOOR_SIZE[2]}',
        'rgba': '0.9 0.9 0.9 1',
        'material': '',
        'condim': '3',
        'friction': '1.0 0.005 0.0001'
    })

    # Add lights
    for i, pos in enumerate(['2 2 3', '-2 2 3', '0 -2 3']):
        ET.SubElement(worldbody, 'light', attrib={
            'name': f'light{i+1}',
            'pos': pos,
            'dir': '0 0 -1',
            'diffuse': '0.8 0.8 0.8',
            'specular': '0.2 0.2 0.2'
        })

    # Convert URDF links and joints
    robot_name = urdf_root.get('name', 'robot')
    base_body = convert_urdf_link(urdf_root, 'base_link', worldbody)

    # Add freejoint to base
    ET.SubElement(base_body, 'freejoint', attrib={'name': 'base_freejoint'})

    # Set base position
    base_body.set('pos', f'0 0 {BASE_INIT_Z}')

    # Recursively convert children
    convert_urdf_tree(urdf_root, 'base_link', base_body)

    return mjcf


def convert_urdf_link(urdf_root: ET.Element, link_name: str, parent_mjcf: ET.Element) -> ET.Element:
    """Convert a URDF link to MJCF body"""
    urdf_link = urdf_root.find(f".//link[@name='{link_name}']")
    if urdf_link is None:
        return None

    # Create MJCF body
    body = ET.SubElement(parent_mjcf, 'body', attrib={'name': link_name})

    # Add inertial
    inertial = urdf_link.find('inertial')
    if inertial is not None:
        mjcf_inertial = ET.SubElement(body, 'inertial')

        # pos is REQUIRED in MuJoCo
        origin = inertial.find('origin')
        if origin is not None:
            pos = origin.get('xyz', '0 0 0')
        else:
            pos = '0 0 0'
        mjcf_inertial.set('pos', pos)

        mass = inertial.find('mass')
        if mass is not None:
            mjcf_inertial.set('mass', mass.get('value'))

        inertia_elem = inertial.find('inertia')
        if inertia_elem is not None:
            diaginertia = f"{inertia_elem.get('ixx')} {inertia_elem.get('iyy')} {inertia_elem.get('izz')}"
            mjcf_inertial.set('diaginertia', diaginertia)

    # Add visual geoms
    for visual in urdf_link.findall('visual'):
        geom = visual.find('geometry')
        origin = visual.find('origin')

        geom_attrib = {'type': 'box'}

        if origin is not None:
            geom_attrib['pos'] = origin.get('xyz', '0 0 0')
            rpy = origin.get('rpy', '0 0 0')
            if rpy != '0 0 0':
                geom_attrib['euler'] = rpy

        # Parse geometry
        if geom is not None:
            box = geom.find('box')
            cylinder = geom.find('cylinder')
            sphere = geom.find('sphere')

            if box is not None:
                size = box.get('size')
                size_vals = [float(x)/2 for x in size.split()]  # MJCF uses half-sizes
                geom_attrib['size'] = f"{size_vals[0]} {size_vals[1]} {size_vals[2]}"
                geom_attrib['type'] = 'box'
            elif cylinder is not None:
                geom_attrib['type'] = 'cylinder'
                geom_attrib['size'] = f"{cylinder.get('radius')} {float(cylinder.get('length'))/2}"
            elif sphere is not None:
                geom_attrib['type'] = 'sphere'
                geom_attrib['size'] = sphere.get('radius')

        # Material
        material = visual.find('material')
        if material is not None:
            mat_name = material.get('name')
            if mat_name:
                geom_attrib['material'] = mat_name

            color = material.find('color')
            if color is not None:
                geom_attrib['rgba'] = color.get('rgba')

        ET.SubElement(body, 'geom', attrib=geom_attrib)

    return body


def convert_urdf_tree(urdf_root: ET.Element, parent_link: str, parent_body: ET.Element):
    """Recursively convert URDF tree to MJCF"""
    # Find all joints with this parent
    for joint in urdf_root.findall('joint'):
        parent = joint.find('parent')
        if parent is None or parent.get('link') != parent_link:
            continue

        child = joint.find('child')
        if child is None:
            continue

        child_link = child.get('link')
        joint_name = joint.get('name')
        joint_type = joint.get('type')

        # Create child body
        child_body = convert_urdf_link(urdf_root, child_link, parent_body)
        if child_body is None:
            continue

        # Set position from joint origin
        origin = joint.find('origin')
        if origin is not None:
            child_body.set('pos', origin.get('xyz', '0 0 0'))
            rpy = origin.get('rpy', '0 0 0')
            if rpy != '0 0 0':
                child_body.set('euler', rpy)

        # Add joint
        axis = joint.find('axis')
        axis_xyz = axis.get('xyz', '1 0 0') if axis is not None else '1 0 0'

        joint_attrib = {
            'name': joint_name,
            'axis': axis_xyz
        }

        # Map URDF joint types to MJCF
        if joint_type == 'revolute':
            joint_attrib['type'] = 'hinge'
            limit = joint.find('limit')
            if limit is not None:
                joint_attrib['range'] = f"{limit.get('lower')} {limit.get('upper')}"
        elif joint_type == 'continuous':
            joint_attrib['type'] = 'hinge'
            joint_attrib['limited'] = 'false'
        elif joint_type == 'prismatic':
            joint_attrib['type'] = 'slide'
            limit = joint.find('limit')
            if limit is not None:
                joint_attrib['range'] = f"{limit.get('lower')} {limit.get('upper')}"
        elif joint_type == 'fixed':
            # Skip fixed joints - just merge bodies
            continue

        # Add dynamics (damping, friction, armature)
        dynamics = joint.find('dynamics')
        if dynamics is not None:
            damping = dynamics.get('damping')
            friction = dynamics.get('friction')
            if damping:
                joint_attrib['damping'] = damping
            if friction:
                joint_attrib['frictionloss'] = friction

        ET.SubElement(child_body, 'joint', attrib=joint_attrib)

        # Recurse
        convert_urdf_tree(urdf_root, child_link, child_body)


def add_actuators(mjcf: ET.Element):
    """Add actuators for all joints"""
    actuator = ET.SubElement(mjcf, 'actuator')

    worldbody = mjcf.find('worldbody')

    # Find all joints
    for joint in worldbody.iter('joint'):
        joint_name = joint.get('name')

        if not joint_name or 'free' in joint_name:
            continue

        # Velocity servos for wheels
        if 'wheel' in joint_name:
            ET.SubElement(actuator, 'velocity', attrib={
                'name': f'vel_{joint_name}',
                'joint': joint_name,
                'kv': '1.0',
                'ctrlrange': '-10 10'
            })
        # Position servos for everything else
        else:
            joint_range = joint.get('range', '-3.14 3.14')
            ET.SubElement(actuator, 'position', attrib={
                'name': f'pos_{joint_name}',
                'joint': joint_name,
                'kp': '100',
                'kv': '10',
                'ctrlrange': joint_range
            })


def add_tcp_sites(mjcf: ET.Element):
    """Add TCP sites"""
    worldbody = mjcf.find('worldbody')
    tcp_offset = load_tool_geometry().get('tcp_offset', [0.0, 0.0, 0.26])

    for side in ['left', 'right']:
        wrist = worldbody.find(f".//body[@name='{side}_link7']")
        if wrist is not None:
            ET.SubElement(wrist, 'site', attrib={
                'name': f'{side}_tcp',
                'pos': ' '.join(str(value) for value in tcp_offset),
                'type': 'sphere',
                'size': '0.01',
                'rgba': '1 0 0 0.5'
            })


def add_grippers(mjcf: ET.Element):
    """Add simple grippers"""
    worldbody = mjcf.find('worldbody')
    actuator = mjcf.find('actuator')
    tool_geometry = load_tool_geometry()
    finger_base_z = float(tool_geometry.get('finger_base_z', 0.10))
    finger_half_size = tool_geometry.get(
        'finger_half_size', [0.006, 0.008, 0.08])
    finger_size = ' '.join(str(value) for value in finger_half_size)

    for side in ['left', 'right']:
        wrist = worldbody.find(f".//body[@name='{side}_link7']")
        if wrist is None:
            continue

        # Left finger
        left_finger = ET.SubElement(wrist, 'body', attrib={
            'name': f'{side}_gripper_left_finger',
            'pos': f'-0.02 0 {finger_base_z}'
        })

        ET.SubElement(left_finger, 'joint', attrib={
            'name': f'{side}_gripper_left_finger_joint',
            'type': 'slide',
            'axis': '-1 0 0',
            'range': '0 0.02',
            'damping': '10'
        })

        ET.SubElement(left_finger, 'geom', attrib={
            'type': 'box',
            'size': finger_size,
            'pos': f'0 0 {finger_half_size[2]}',
            'rgba': '0.3 0.3 0.3 1',
            'friction': '2.0 0.005 0.0001'
        })

        # Right finger
        right_finger = ET.SubElement(wrist, 'body', attrib={
            'name': f'{side}_gripper_right_finger',
            'pos': f'0.02 0 {finger_base_z}'
        })

        ET.SubElement(right_finger, 'joint', attrib={
            'name': f'{side}_gripper_right_finger_joint',
            'type': 'slide',
            'axis': '1 0 0',
            'range': '0 0.02',
            'damping': '10'
        })

        ET.SubElement(right_finger, 'geom', attrib={
            'type': 'box',
            'size': finger_size,
            'pos': f'0 0 {finger_half_size[2]}',
            'rgba': '0.3 0.3 0.3 1',
            'friction': '2.0 0.005 0.0001'
        })

        # Actuator
        ET.SubElement(actuator, 'position', attrib={
            'name': f'{side}_gripper',
            'joint': f'{side}_gripper_left_finger_joint',
            'kp': '50',
            'kv': '5',
            'ctrlrange': '0 0.02'
        })

    # Add equality constraints for finger coupling
    equality = ET.SubElement(mjcf, 'equality')
    for side in ['left', 'right']:
        ET.SubElement(equality, 'joint', attrib={
            'name': f'{side}_gripper_couple',
            'joint1': f'{side}_gripper_left_finger_joint',
            'joint2': f'{side}_gripper_right_finger_joint',
            'polycoef': '0 1 0 0 0'
        })


def prettify(elem, level=0):
    """Add indentation"""
    indent = "  "
    i = "\n" + level * indent
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            prettify(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def main():
    print("=" * 60)
    print("Simplified MJCF Builder (No MuJoCo required)")
    print("=" * 60)

    print(f"\nReading URDF: {URDF_PATH}")
    urdf_root = parse_urdf(URDF_PATH)

    print("Converting URDF to MJCF...")
    mjcf = urdf_to_mjcf(urdf_root)

    print("Adding actuators...")
    add_actuators(mjcf)

    print("Adding TCP sites...")
    add_tcp_sites(mjcf)

    print("Adding grippers...")
    add_grippers(mjcf)

    print(f"\nWriting MJCF: {OUTPUT_PATH}")
    prettify(mjcf)
    tree = ET.ElementTree(mjcf)
    tree.write(OUTPUT_PATH, encoding='utf-8', xml_declaration=True)

    print("\n" + "=" * 60)
    print("Build complete!")
    print("=" * 60)
    print(f"\nTo view (requires mujoco installed):")
    print(f"  python -m mujoco.viewer --mjcf={OUTPUT_PATH}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
