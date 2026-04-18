import re

with open("lerobot/scripts/lerobot_flexiv_teleop_record.py", "r") as f:
    code = f.read()

# 1. Strip out obsolete consolidate
code = code.replace('dataset.consolidate()', '')

# 2. Fix Leader IK mapping and Relative framework
import_start = code.find('def get_target_joints(self, dynamixel_joints, current_flexiv_joints=None):')
import_end = code.find('elif self.robot_type == "so100":')

relative_code = """def get_target_joints(self, dynamixel_joints, current_flexiv_joints=None):
        \"\"\"Calculates IK to return 7 joint angles for Rizon 4.\"\"\"
        if self.robot_type == "koch":
            # LeRobot 3.0 provides zero-centered, calibrated dynamixel joints.
            # No 90-degree offsets are needed! Applying offsets breaks the geometry.
            data = [
                -dynamixel_joints[0],
                dynamixel_joints[1],
                dynamixel_joints[2],
                dynamixel_joints[3],
                dynamixel_joints[4],
                dynamixel_joints[5],
            ]
            data = [angle * np.pi / 180.0 for angle in data]
            for i, joint in enumerate(data[:6]):
                pb.resetJointState(self.robot_pb, i, joint)
            
            eef_pos = np.array(pb.getLinkState(self.robot_pb, 4)[0])
            new_eef_orn = pb.getLinkState(self.robot_pb, 4)[1]
            
            # Simple Gripper Mapping
            raw_gripper = dynamixel_joints[5]
            gripper_normalized = max(0.0, min(1.0, abs(raw_gripper) / 90.0))
            gripper = gripper_normalized * 0.1

        """
code = code[:import_start] + relative_code + code[import_end:]

# 3. Implement Relative Math at the bottom
relative_math_start = code.find('# We are performing ABSOLUTE 1:1 Workspace tracking.')
relative_math_end = code.find('joints = list(joints_tuple[:7])')

relative_math_code = """# --- RELATIVE EEF TRACKING ---
        target_link_index = 7 # flange
        
        if self.init_leader_pos is None:
            self.init_leader_pos = eef_pos
            self.init_leader_orn = new_eef_orn
            
            # Sync to physical Flexiv on first frame to capture true hardware EEF
            if current_flexiv_joints is not None:
                for i, jv in enumerate(current_flexiv_joints):
                    pb.resetJointState(self.follow_arm, i, float(jv))
                    
            state = pb.getLinkState(self.follow_arm, target_link_index)
            self.init_follower_pos = np.array(state[0])
            self.init_follower_orn = state[1]

        # Compute translation delta scaled by 4.2
        delta_pos = (eef_pos - self.init_leader_pos) * 4.2
        target_pos = self.init_follower_pos + delta_pos
        
        # Compute orientation delta using pybullet matrix math: delta_orn = curr_leader * inv(init_leader)
        inv_init_leader_pos, inv_init_leader_orn = pb.invertTransform([0,0,0], self.init_leader_orn)
        _, delta_orn = pb.multiplyTransforms([0,0,0], new_eef_orn, [0,0,0], inv_init_leader_orn)
        
        # Target orn = delta_orn * init_follower_orn
        _, target_orn = pb.multiplyTransforms([0,0,0], delta_orn, [0,0,0], self.init_follower_orn)

        joints_tuple = pb.calculateInverseKinematics(
            self.follow_arm, target_link_index, target_pos, target_orn
        )
        
        """

code = code[:relative_math_start] + relative_math_code + code[relative_math_end:]

with open("lerobot/scripts/lerobot_flexiv_teleop_record.py", "w") as f:
    f.write(code)
