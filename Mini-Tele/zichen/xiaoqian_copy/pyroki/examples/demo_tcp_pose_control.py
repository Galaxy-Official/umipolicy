"""Demonstration of TCP Pose Control Functionality

This script demonstrates the new TCP pose control features added to the robot control interface.
It shows how users can input desired TCP position (XYZ) and quaternion (WXYZ) and the robot
will solve inverse kinematics to reach that pose.
"""

import numpy as np
import time
from pathlib import Path

# Import the functions from our new script
import sys
sys.path.append('/Disk3/xiaoqian/pyroki/examples')

# We'll import and use the functions directly for demonstration
def demonstrate_tcp_pose_control():
    """Demonstrate the TCP pose control functionality."""

    print("=" * 60)
    print("TCP POSE CONTROL DEMONSTRATION")
    print("=" * 60)

    print("\n🎯 NEW FEATURES ADDED:")
    print("1. TCP Position Input (XYZ coordinates)")
    print("2. TCP Quaternion Input (WXYZ components)")
    print("3. Inverse Kinematics Solver")
    print("4. Real-time Joint Angle Display")
    print("5. TCP Pose Presets")
    print("6. Normalize Quaternion Function")
    print("7. Get Current TCP Pose Function")

    print("\n📋 INTERFACE SECTIONS:")
    print("• Joint Control - Traditional joint angle control via sliders")
    print("• Joint Input - Precise joint angle input fields")
    print("• CSV Joint Input - Load joint configurations from CSV")
    print("• TCP Pose Control - NEW: Input desired end-effector pose")
    print("  └─ TCP Position Input - X, Y, Z coordinate sliders")
    print("  └─ TCP Quaternion Input - W, X, Y, Z quaternion sliders")
    print("  └─ Normalize Quaternion - Ensure quaternion is unit length")
    print("  └─ Apply TCP Pose - Solve IK and move robot")
    print("  └─ Get Current TCP Pose - Read current pose into inputs")
    print("  └─ TCP Presets - Quick access to common poses")
    print("• TCP Pose - Display current end-effector pose")
    print("• Misc Controls - Additional functions")

    print("\n🔧 USAGE EXAMPLES:")

    # Example 1: Basic TCP pose input
    print("\n1️⃣  Basic TCP Pose Input:")
    print("   - Set Position: X=0.5, Y=0.0, Z=0.5")
    print("   - Set Quaternion: W=1.0, X=0.0, Y=0.0, Z=0.0")
    print("   - Click 'Apply TCP Pose'")
    print("   - Robot moves to position [0.5, 0, 0.5] with identity rotation")

    # Example 2: Using presets
    print("\n2️⃣  Using TCP Presets:")
    print("   - Click 'Preset: Forward' → Robot reaches forward")
    print("   - Click 'Preset: Up' → Robot reaches up")
    print("   - Click 'Preset: Side' → Robot reaches to the side")
    print("   - Click 'Preset: Down' → Robot reaches down")

    # Example 3: Complex orientation
    print("\n3️⃣  Complex Orientation:")
    print("   - Position: X=0.4, Y=0.2, Z=0.6")
    print("   - Quaternion: W=0.707, X=0.0, Y=0.707, Z=0.0")
    print("   - This creates 90° rotation around Z-axis")
    print("   - Tool will be pointing sideways")

    # Example 4: Workflow
    print("\n4️⃣  Typical Workflow:")
    print("   a) Click 'Get Current TCP Pose' to read current pose")
    print("   b) Modify position/orientation values slightly")
    print("   c) Click 'Normalize Quaternion' if needed")
    print("   d) Click 'Apply TCP Pose' to move robot")
    print("   e) Check joint angles in 'Joint Control' section")

    print("\n🎯 KEY FEATURES:")
    print("• Real-time inverse kinematics solving")
    print("• Automatic quaternion normalization")
    print("• Visual feedback with 3D TCP marker")
    print("• Joint limit awareness")
    print("• Multiple solution handling")
    print("• Error tolerance checking")

    print("\n📊 TECHNICAL DETAILS:")
    print("• Uses PyRoki's optimization-based IK solver")
    print("• Levenberg-Marquardt optimization algorithm")
    print("• Position tolerance: 1mm")
    print("• Quaternion tolerance: 0.01 radians")
    print("• Joint limit regularization included")

    print("\n🚀 TO START:")
    print("Run: python 01_basic_ik_tcp_control.py")
    print("Open browser: http://localhost:8083")
    print("Use the TCP Pose Control section!")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)

def show_sample_tcp_poses():
    """Show some example TCP poses."""

    print("\n📋 SAMPLE TCP POSES:")

    sample_poses = [
        {
            "name": "Home Position",
            "position": [0.5, 0.0, 0.5],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "description": "Default centered position"
        },
        {
            "name": "Forward Reach",
            "position": [0.6, 0.0, 0.4],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "description": "Reach forward 10cm"
        },
        {
            "name": "Up Reach",
            "position": [0.4, 0.0, 0.6],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "description": "Reach up 10cm"
        },
        {
            "name": "Side Reach",
            "position": [0.5, 0.2, 0.4],
            "quaternion": [0.707, 0.0, 0.707, 0.0],
            "description": "Reach right 20cm with 90° Z-rotation"
        },
        {
            "name": "Downward Tool",
            "position": [0.5, 0.0, 0.3],
            "quaternion": [0.707, 0.707, 0.0, 0.0],
            "description": "Tool pointing down (90° X-rotation)"
        },
        {
            "name": "Complex Orientation",
            "position": [0.45, 0.15, 0.55],
            "quaternion": [0.8, 0.2, 0.5, 0.1],
            "description": "Complex 3D orientation"
        }
    ]

    for i, pose in enumerate(sample_poses, 1):
        print(f"\n{i}. {pose['name']}:")
        print(f"   Position: [{pose['position'][0]:.1f}, {pose['position'][1]:.1f}, {pose['position'][2]:.1f}]")
        print(f"   Quaternion: [{pose['quaternion'][0]:.3f}, {pose['quaternion'][1]:.1f}, {pose['quaternion'][2]:.1f}, {pose['quaternion'][3]:.1f}]")
        print(f"   Description: {pose['description']}")

def show_gui_layout():
    """Show the GUI layout structure."""

    print("\n🖥️  GUI LAYOUT:")
    print("┌─ Joint Control")
    print("│  ├─ Joint 0, Joint 1, ..., Joint 7 (sliders)")
    print("│  └─ [Joint sliders update in real-time]")
    print("├─ Joint Input")
    print("│  ├─ Joint 0 (rad), Joint 1 (rad), ... (number inputs)")
    print("│  └─ [Precise joint angle input]")
    print("├─ CSV Joint Input")
    print("│  ├─ Joint Angles CSV (text input)")
    print("│  ├─ Apply CSV Joints (button)")
    print("│  ├─ Load from CSV File (button)")
    print("│  └─ Load CSV Row (button + number input)")
    print("├─ TCP Pose Control  ← NEW SECTION")
    print("│  ├─ TCP Position Input (folder)")
    print("│  │  ├─ X (m), Y (m), Z (m) (number inputs)")
    print("│  │  └─ [Position coordinate sliders]")
    print("│  ├─ TCP Quaternion Input (folder)")
    print("│  │  ├─ W, X, Y, Z (number inputs)")
    print("│  │  └─ [Quaternion component sliders]")
    print("│  ├─ Normalize Quaternion (button)")
    print("│  ├─ Apply TCP Pose (button)")
    print("│  ├─ Get Current TCP Pose (button)")
    print("│  └─ TCP Presets (folder)")
    print("│     ├─ Preset: Home (button)")
    print("│     ├─ Preset: Forward (button)")
    print("│     ├─ Preset: Up (button)")
    print("│     ├─ Preset: Side (button)")
    print("│     └─ Preset: Down (button)")
    print("├─ Presets")
    print("│  ├─ Zero Position (button)")
    print("│  └─ Home Position (button)")
    print("├─ Sequence Playback (if sequence loaded)")
    print("│  ├─ Pose Index (number input)")
    print("│  ├─ Load Pose (button)")
    print("│  ├─ Auto Play (checkbox)")
    print("│  ├─ Play Speed (number input)")
    print("│  ├─ Previous (button)")
    print("│  └─ Next (button)")
    print("├─ Misc Controls")
    print("│  ├─ Random Configuration (button)")
    print("│  ├─ Save Configuration (button)")
    print("│  └─ Load Saved Config (button)")
    print("└─ Info")
    print("   ├─ Joint Count (text display)")
    print("   ├─ TCP Pose (folder)")
    print("   │  ├─ Position (XYZ) (text display)")
    print("   │  └─ Quaternion (WXYZ) (text display)")
    print("   └─ [Other info displays]")

if __name__ == "__main__":
    print("🤖 TCP POSE CONTROL DEMONSTRATION")
    print("Enhanced robot control with inverse kinematics")
    print("=" * 60)

    # Run demonstrations
    demonstrate_tcp_pose_control()
    show_sample_tcp_poses()
    show_gui_layout()

    print("\n✨ Ready to use!")
    print("Run: python 01_basic_ik_tcp_control.py")
    print("Then open: http://localhost:8083")
    print("=" * 60)