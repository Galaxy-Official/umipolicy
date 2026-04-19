import re

with open("/home/rhos/umipolicy/lerobot/src/lerobot/scripts/lerobot_flexiv_teleop_record.py", "r") as f:
    text = f.read()

ws_block = """
def is_eef_within_workspace(eef_pos) -> bool:
    import numpy as np
    if eef_pos is None:
        return True

    lowcost_lower_bound = np.array([-0.09, 0.09, 0.02])
    lowcost_upper_bound = np.array([0.06, 0.24, 0.16])
    return not (np.any(eef_pos > lowcost_upper_bound) or np.any(eef_pos < lowcost_lower_bound))
"""

text = ws_block + "\n" + text

# Locate the fk_resolver.get_target_joints block in teleop loop
loop_pattern = r"(target_joints, _t_gripper = fk_resolver\.get_target_joints.*?)(action_state = np\.concatenate.*?env\.exec_action_joints\(target_joints=target_joints, target_width=_t_gripper\))"

# We must extract eef_pos from fk_resolver
# Let's cleanly modify get_target_joints to return eef_pos as well
fk_sig = r"return joints, gripper"
fk_new = r"return joints, gripper, eef_pos"
text = re.sub(fk_sig, fk_new, text)

exec_logic = """
            target_joints, _t_gripper, eef_pos = fk_resolver.get_target_joints(raw_ticks, current_flexiv_joints=arm_state)
            
            if not is_eef_within_workspace(eef_pos):
                logger.warning(f"Out of bounds! Blocking motion: {eef_pos}")
                continue
                
            action_state = np.concatenate([target_joints, np.array([_t_gripper])])
            
            # Exec Flexiv Joint Position
            env.exec_action_joints(target_joints=target_joints, target_width=_t_gripper)
"""
text = re.sub(r"target_joints, _t_gripper = fk_resolver.*?env\.exec_action_joints\(target_joints=target_joints, target_width=_t_gripper\)", exec_logic, text, flags=re.DOTALL)

with open("/home/rhos/umipolicy/lerobot/src/lerobot/scripts/lerobot_flexiv_teleop_record.py", "w") as f:
    f.write(text)

print("Safety patched.")
