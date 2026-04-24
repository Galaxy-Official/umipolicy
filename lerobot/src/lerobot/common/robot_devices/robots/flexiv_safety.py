import numpy as np
import scipy.spatial.transform as st
import math
from loguru import logger

# Strict Cartesian Limits based on user calibration
X_MIN = 0.2990
X_MAX = 1.0130
Y_MIN = -0.4330
Y_MAX = 0.3680
Z_MIN_BASE = 0.0715
Z_MAX = 0.6360

# Safe margin (in meters) added to Z_MIN when the gripper rotates.
# Assuming a 5cm buffer for the gripper finger edge hitting the table when rolled 90 degrees.
Z_ROT_MARGIN = 0.035

def get_safe_z_min(roll_angle):
    """Calculate dynamic Z boundary based on Roll angle"""
    return Z_MIN_BASE + Z_ROT_MARGIN * abs(math.sin(roll_angle))

def _check_and_warn(x, y, z, safe_x, safe_y, safe_z):
    if abs(x - safe_x) > 1e-4 or abs(y - safe_y) > 1e-4 or abs(z - safe_z) > 1e-4:
        logger.opt(colors=True).warning(
            f"<red>[Safety Boundary Triggered] Target: ({x:.4f}, {y:.4f}, {z:.4f}) -> Clipped: ({safe_x:.4f}, {safe_y:.4f}, {safe_z:.4f})</red>"
        )

def clip_target_pose_10d(target_pose_10d):
    """
    Clips a 10D Cartesian pose [x, y, z, rx, ry, rz, width, ...]
    """
    x, y, z = target_pose_10d[0], target_pose_10d[1], target_pose_10d[2]
    rx, ry, rz = target_pose_10d[3], target_pose_10d[4], target_pose_10d[5]
    
    try:
        # Determine Roll
        rot = st.Rotation.from_rotvec([rx, ry, rz])
        roll = rot.as_euler('xyz')[0]
        z_min = get_safe_z_min(roll)
    except Exception:
        z_min = Z_MIN_BASE + Z_ROT_MARGIN
        
    safe_x = np.clip(x, X_MIN, X_MAX)
    safe_y = np.clip(y, Y_MIN, Y_MAX)
    safe_z = np.clip(z, z_min, Z_MAX)
    
    _check_and_warn(x, y, z, safe_x, safe_y, safe_z)
    
    target_pose_10d[0] = safe_x
    target_pose_10d[1] = safe_y
    target_pose_10d[2] = safe_z
    return target_pose_10d

def clip_target_pose_7d(target_tcp):
    """
    Clips a 7D Cartesian pose [x, y, z, qw, qx, qy, qz]
    """
    x, y, z = target_tcp[0], target_tcp[1], target_tcp[2]
    qw, qx, qy, qz = target_tcp[3], target_tcp[4], target_tcp[5], target_tcp[6]
    
    try:
        # Determine Roll
        rot = st.Rotation.from_quat([qx, qy, qz, qw], scalar_first=False)
        roll = rot.as_euler('xyz')[0]
        z_min = get_safe_z_min(roll)
    except Exception:
        z_min = Z_MIN_BASE + Z_ROT_MARGIN

    safe_x = float(np.clip(x, X_MIN, X_MAX))
    safe_y = float(np.clip(y, Y_MIN, Y_MAX))
    safe_z = float(np.clip(z, z_min, Z_MAX))
    
    _check_and_warn(x, y, z, safe_x, safe_y, safe_z)
    
    target_tcp[0] = safe_x
    target_tcp[1] = safe_y
    target_tcp[2] = safe_z
    return target_tcp

def is_eef_within_workspace_strict(eef_pos):
    """
    Checks if a 3D eef position is within the strict boundaries.
    Ignores dynamic Z margin because orientation is not passed here.
    """
    if eef_pos is None:
        return True
    
    x, y, z = eef_pos[0], eef_pos[1], eef_pos[2]
    
    if not (X_MIN <= x <= X_MAX): return False
    if not (Y_MIN <= y <= Y_MAX): return False
    if not (Z_MIN_BASE <= z <= Z_MAX): return False
    
    return True
