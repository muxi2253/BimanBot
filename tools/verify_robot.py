#!/usr/bin/env python3
"""验证机器人 MJCF 能否正确加载"""

import mujoco
import sys

def main():
    xml_path = 'robot/dual_arm_robot.xml'

    print(f"Loading: {xml_path}")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        print("✓ Model loaded successfully!")
        print(f"\nModel statistics:")
        print(f"  Bodies: {model.nbody}")
        print(f"  Joints: {model.njnt}")
        print(f"  DOF: {model.nv}")
        print(f"  Actuators: {model.nu}")
        print(f"  Geoms: {model.ngeom}")

        # List actuators
        print(f"\nActuators:")
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  [{i}] {name}")

        # Check for required sites
        print(f"\nTCP Sites:")
        for side in ['left', 'right']:
            tcp_name = f'{side}_tcp'
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_name)
            if site_id >= 0:
                print(f"  ✓ {tcp_name}")
            else:
                print(f"  ✗ {tcp_name} NOT FOUND")

        return 0

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
