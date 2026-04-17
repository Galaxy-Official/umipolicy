"""Test CSV input functionality"""

import numpy as np

# Test the specific CSV data you mentioned
csv_data = "0.780475,-0.100728,0.294434,-0.426711,-0.068814,-0.034665,-0.901100"

print("Testing CSV parsing with your data:")
print(f"Input: {csv_data}")

try:
    # Parse CSV input
    joint_values = [float(x.strip()) for x in csv_data.split(',')]
    print(f"Parsed {len(joint_values)} joint values: {joint_values}")

    # Check ranges
    for i, val in enumerate(joint_values):
        print(f"Joint {i}: {val:.6f} rad ({np.degrees(val):.2f} deg)")

    print("CSV parsing successful!")

except ValueError as e:
    print(f"Error parsing CSV: {e}")
except Exception as e:
    print(f"Error: {e}")

# Test clamping function
print("\nTesting joint clamping:")
test_values = [0.780475, -0.100728, 0.294434, -0.426711, -0.068814, -0.034665, -0.901100]
clamped_values = [np.clip(val, -np.pi, np.pi) for val in test_values]

print("Original values:", [f"{v:.6f}" for v in test_values])
print("Clamped values:  ", [f"{v:.6f}" for v in clamped_values])

# Check if any values were clamped
for i, (orig, clamp) in enumerate(zip(test_values, clamped_values)):
    if orig != clamp:
        print(f"Joint {i} was clamped from {orig:.6f} to {clamp:.6f}")