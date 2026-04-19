import re

with open("/home/rhos/umipolicy/lerobot/src/lerobot/scripts/lerobot_flexiv_teleop_record.py", "r") as f:
    text = f.read()

original_fk = """    def __init__(self, robot_type="koch"):
        self.robot_type = robot_type
        # Pybullet GUI overhead avoidance
        pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        URDF_DICT = {
            "koch": "urdf/assets/low_cost_robot_description/urdf/low_cost_robot.urdf",
            "so100": "urdf/assets/SO_5DOF_ARM100_8j_URDF.SLDASM/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf",
        }
        urdf_path = URDF_DICT.get(self.robot_type, None)
        if urdf_path is None:
            raise ValueError(f"Unknown URDF for {self.robot_type}")
            
        base_orien = [0, 0, 1, 0] if self.robot_type == "so100" else [0, 0, 0, 1]
        self.robot_pb = pb.loadURDF(
            urdf_path,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=base_orien,
            useFixedBase=True,
        )

        flexiv_urdf_path = "urdf/assets/rizon/flexiv_rizon4.urdf"
        self.follow_arm = pb.loadURDF(
            flexiv_urdf_path,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0, 0, 0, 1],  # Standard base (removed 90 degree sideways mount from old leap hand)
            useFixedBase=True,
        )
        
        # Relative Tracking States
        self.init_leader_pos = None
        self.init_leader_orn = None
        self.init_follower_pos = None
        self.init_follower_orn = None

    def get_target_joints(self, dynamixel_joints, current_flexiv_joints=None):
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

        elif self.robot_type == "so100":
            data = [
                -dynamixel_joints[0],
                90 - dynamixel_joints[1],
                dynamixel_joints[2] - 90,
                dynamixel_joints[3] - 90,
                90 - dynamixel_joints[4],
                dynamixel_joints[5],
            ]
            data = [angle * np.pi / 180 for angle in data]
            for i, joint in enumerate(data):
                pb.resetJointState(self.robot_pb, i, joint)
            
            eef_pos = np.array(pb.getLinkState(self.robot_pb, 4)[0])
            eef_orn = pb.getLinkState(self.robot_pb, 4)[1]
            
            eef_orn_matrix = np.array(pb.getMatrixFromQuaternion(eef_orn)).reshape(3, 3)
            original_axes = np.eye(3)
            z_in_eef = np.array([-1, 0, 0])
            x_in_eef = np.array([0, -1, 0])
            y_in_eef = np.array([0, 0, 1])
            new_axes = np.array([x_in_eef, y_in_eef, z_in_eef]).T
            new_axes = np.concatenate([new_axes, np.array([[1, 1, 1]])], axis=0)
            
            x_ab = np.array([np.dot(eef_orn_matrix[:, 0], original_axes[i]) for i in range(3)])
            y_ab = np.array([np.dot(eef_orn_matrix[:, 1], original_axes[i]) for i in range(3)])
            z_ab = np.array([np.dot(eef_orn_matrix[:, 2], original_axes[i]) for i in range(3)])
            
            T_ab = np.array([x_ab, y_ab, z_ab, eef_pos]).T
            T_ab = np.concatenate([T_ab, np.array([[0, 0, 0, 1]])], axis=0)
            rot_axes = (T_ab @ new_axes)[:3]
            
            def matrix2quaternion(matrix):
                tr = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
                if tr > 0:
                    S = np.sqrt(tr + 1.0) * 2
                    qw = 0.25 * S
                    qx = (matrix[2, 1] - matrix[1, 2]) / S
                    qy = (matrix[0, 2] - matrix[2, 0]) / S
                    qz = (matrix[1, 0] - matrix[0, 1]) / S
                elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
                    S = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
                    qw = (matrix[2, 1] - matrix[1, 2]) / S
                    qx = 0.25 * S
                    qy = (matrix[0, 1] + matrix[1, 0]) / S
                    qz = (matrix[0, 2] + matrix[2, 0]) / S
                elif matrix[1, 1] > matrix[2, 2]:
                    S = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
                    qw = (matrix[0, 2] - matrix[2, 0]) / S
                    qx = (matrix[0, 1] + matrix[1, 0]) / S
                    qy = 0.25 * S
                    qz = (matrix[1, 2] + matrix[2, 1]) / S
                else:
                    S = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
                    qw = (matrix[1, 0] - matrix[0, 1]) / S
                    qx = (matrix[0, 2] + matrix[2, 0]) / S
                    qy = (matrix[1, 2] + matrix[2, 1]) / S
                    qz = 0.25 * S
                return np.array([qx, qy, qz, qw])
            new_eef_orn = matrix2quaternion(rot_axes)
            gripper = data[5]

        # --- RELATIVE EEF TRACKING ---
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
        
        joints = list(joints_tuple[:7])
        for i, joint in enumerate(joints):
            # Advance PyBullet's internal solver
            pb.resetJointState(self.follow_arm, i, joint)
            
        return joints, gripper"""

text = re.sub(r"    def __init__\(self, robot_type=\"koch\"\):.*?return joints, gripper", original_fk, text, flags=re.DOTALL)
with open("/home/rhos/umipolicy/lerobot/src/lerobot/scripts/lerobot_flexiv_teleop_record.py", "w") as f:
    f.write(text)
print("Reverted!")
