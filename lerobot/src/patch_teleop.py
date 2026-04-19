import re

with open("/home/rhos/umipolicy/lerobot/src/lerobot/scripts/lerobot_flexiv_teleop_record.py", "r") as f:
    text = f.read()

# 1. Update `LeaderFKResolver.get_target_joints()` signature and logic
fk_pattern = r"(def get_target_joints\(self, .*?current_flexiv_joints=None\):).*?(elif self\.robot_type == \"so100\":)"
fk_old = re.search(fk_pattern, text, re.DOTALL).group(0)

fk_new = """def get_target_joints(self, raw_ticks, current_flexiv_joints=None):
        \"\"\"Calculates IK to return 7 joint angles for Rizon 4.\"\"\"
        if self.robot_type == "koch":
            # LeRobot 2.0 legacy tracking geometry using raw encoder ticks
            data = [
                - (raw_ticks["shoulder_pan"] - 2048) * (3.14159 / 2048.0),
                  (raw_ticks["shoulder_lift"] - 2048) * (3.14159 / 2048.0),
                  (raw_ticks["elbow_flex"] - 2048) * (3.14159 / 2048.0),
                  (raw_ticks["wrist_flex"] - 2048) * (3.14159 / 2048.0),
                  (raw_ticks["wrist_roll"] - 2048) * (3.14159 / 2048.0),
                - (raw_ticks["gripper"] - 2048) * (3.14159 / 2048.0),
            ]
            for i, joint in enumerate(data[:6]):
                pb.resetJointState(self.robot_pb, i, joint)
            
            eef_pos = np.array(pb.getLinkState(self.robot_pb, 4)[0])
            new_eef_orn = pb.getLinkState(self.robot_pb, 4)[1]
            
            # Gripper Mapping from full continuous bounds
            gripper_tick = raw_ticks["gripper"]
            # 2150 is rest (open), squeezing pushes it away. Assume bounds 1800 to 2200 for open/close mapping.
            # Use raw trigger reading: smaller ticks closer to 2048 means squeeze.
            # But let's reuse simple proportion:
            norm_val = max(0.0, min(1.0, abs(gripper_tick - 2150) / 400.0))
            gripper = (1.0 - norm_val) * 0.1

        elif self.robot_type == "so100":"""

text = text.replace(fk_old, fk_new)

# 2. Update `teleop_step()` to pass `raw_ticks` instead of `dynamixel_joints`
step_pattern = r"dynamixel_joints = \[leader_action\[f\"\{motor\}\.pos\"\] for motor in leader_action_features_to_motors\]\s+target_joints, target_width = fk_resolver\.get_target_joints\(dynamixel_joints, current_flexiv_joints\)"

step_new = """raw_ticks = leader.bus.sync_read("Present_Position", normalize=False)
        target_joints, target_width = fk_resolver.get_target_joints(raw_ticks, current_flexiv_joints)"""

text = re.sub(step_pattern, step_new, text)

print("Patching...")
with open("/home/rhos/umipolicy/lerobot/src/lerobot/scripts/lerobot_flexiv_teleop_record.py", "w") as f:
    f.write(text)
print("Done!")
