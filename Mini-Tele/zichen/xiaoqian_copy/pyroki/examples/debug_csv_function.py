"""Debug CSV input functionality"""

import numpy as np

# Test the exact CSV data you mentioned
csv_data = "0.780475,-0.100728,0.294434,-0.426711,-0.068814,-0.034665,-0.901100"

print("=== Testing CSV Input Functionality ===")
print(f"Input CSV: {csv_data}")

try:
    # Step 1: Parse CSV
    joint_values = [float(x.strip()) for x in csv_data.split(',')]
    print(f"✓ Parsed {len(joint_values)} values: {joint_values}")

    # Step 2: Check robot joint count (8 for Panda)
    num_robot_joints = 8
    num_csv_joints = len(joint_values)
    print(f"✓ CSV joints: {num_csv_joints}, Robot joints: {num_robot_joints}")

    # Step 3: Apply to joints (handle different lengths)
    print("\nApplying to robot joints:")
    for i in range(min(num_csv_joints, num_robot_joints)):
        # Clamp values to valid range
        value = np.clip(joint_values[i], -np.pi, np.pi)
        print(f"  Joint {i}: {joint_values[i]:.6f} -> {value:.6f} (clamped: {joint_values[i] != value})")

    # Step 4: Zero out remaining joints
    for i in range(num_csv_joints, num_robot_joints):
        print(f"  Joint {i}: 0.000000 (default)")

    # Step 5: Create final joint array
    final_joints = np.zeros(num_robot_joints)
    for i in range(min(num_csv_joints, num_robot_joints)):
        final_joints[i] = np.clip(joint_values[i], -np.pi, np.pi)

    print(f"\n✓ Final joint array: {final_joints}")

    # Step 6: Check for potential issues
    print("\nChecking for potential issues:")
    any_nan = np.any(np.isnan(final_joints))
    any_inf = np.any(np.isinf(final_joints))
    print(f"  Contains NaN: {any_nan}")
    print(f"  Contains Inf: {any_inf}")
    print(f"  All finite: {np.all(np.isfinite(final_joints))}")

    print("\n=== Test completed successfully! ===")

except ValueError as e:
    print(f"✗ Error parsing CSV: {e}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test the specific values from your CSV
print(f"\n=== Detailed Analysis ===")
print("Joint angles in degrees:")
for i, val in enumerate(joint_values):
    print(f"  Joint {i}: {val:.6f} rad = {np.degrees(val):.2f} deg")

# Check if values are reasonable for Panda robot
print(f"\nRange analysis (Panda typical range ±π rad):")
for i, val in enumerate(joint_values):
    is_in_range = -np.pi <= val <= np.pi
    print(f"  Joint {i}: {val:.6f} rad - {'✓ OK' if is_in_range else '✗ Out of range'}")