#!/usr/bin/env python3
"""
Add joint damping to all revolute/prismatic joints in URDF
This improves simulation stability
"""

import re
import sys

def add_damping_to_urdf(urdf_path):
    with open(urdf_path, 'r') as f:
        content = f.read()

    # Pattern: joint with limit but no dynamics
    # Match: <limit .../> followed by </joint> without <dynamics> in between
    pattern = r'(<joint[^>]*type="(?:revolute|prismatic)"[^>]*>.*?<limit[^>]*/>)((?!\s*<dynamics).*?)(</joint>)'

    def replacement(match):
        before = match.group(1)
        middle = match.group(2)
        after = match.group(3)

        # Check if dynamics already exists
        if '<dynamics' in middle:
            return match.group(0)

        # Add dynamics after limit
        return before + '\n    <dynamics damping="1.0" friction="0.1"/>' + middle + after

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Count changes
    original_dynamics = content.count('<dynamics')
    new_dynamics = new_content.count('<dynamics')

    with open(urdf_path, 'w') as f:
        f.write(new_content)

    print(f"Added {new_dynamics - original_dynamics} <dynamics> tags to {urdf_path}")
    return new_dynamics - original_dynamics

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python add_joint_damping.py <urdf_file>")
        sys.exit(1)

    count = add_damping_to_urdf(sys.argv[1])
    print(f"Done! Added damping to {count} joints")
