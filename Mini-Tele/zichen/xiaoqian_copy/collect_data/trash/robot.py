import time
import numpy as np
import torch
import transformations as tf
import os
import json
import sys
import threading

from flexiv import *
from utilities.pose_util import *
from utilities.flexivutils import parse_pt_states

sys.path.append('flexiv_api/lib_py')
import flexivrdk


class Flexiv():
    DOF = 7
    TARGET_VEL = [0.0] * DOF
    TARGET_ACC = [0.0] * DOF
    MAX_VEL = [0.1] * DOF
    MAX_ACC = [0.2] * DOF
    
    # # flexiv-1
    ROBOT_IP = "192.168.2.100"
    LOCAL_IP = "192.168.2.103"

    # flexiv-2
    # ROBOT_IP = "192.168.2.100"
    # LOCAL_IP = "192.168.2.102"
    
    tx_flange_tip = np.identity(4)
    tx_flange_tip[:3, 3] = np.array([0, 0, 0.17])  # measured physically
    tx_tip_flange = np.linalg.inv(tx_flange_tip)

    @staticmethod
    def tip_to_flange_pose(tip_pose):
        return mat_to_pose(pose_to_mat(tip_pose) @ Flexiv.tx_tip_flange)


    def __init__(self, robot_ip = ROBOT_IP, local_ip = LOCAL_IP, move_home=True, init_offset=None, init_qpos=None, device="cuda"):
        self.log = log = flexivrdk.Log()
        self.mode = flexivrdk.Mode

        self.sim_rizon = sim_rizon = Rizon4(headless=True)
        self.robot = robot = flexivrdk.Robot(robot_ip, local_ip)
        self.robot_states = robot_states = flexivrdk.RobotStates()
        self.gripper = flexivrdk.Gripper(robot)
        print(self.gripper)
        self.gripper_states = flexivrdk.GripperStates()


        self.verbose = os.environ.get("FLEXIV_=VERBOSE", "0") == "1"

        flange_lower_limits, flange_upper_limits = sim_rizon.get_flange_limits()
        flange_lower_limits = torch.from_numpy(flange_lower_limits).to(device)
        flange_upper_limits = torch.from_numpy(flange_upper_limits).to(device)

        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear ...")
            robot.clearFault()
            time.sleep(2)
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        log.info("Enabling robot ...")
        robot.enable()
        while not robot.isOperational():
            time.sleep(1)

        self.last_send_pose = None

        # =============== init pose =========================

        # assert init_offset is None or init_qpos is None, "Only one of init_offset or init_qpos can be provided"
        assert not (init_offset is not None and not move_home), "init_offset is only valid when move_home is True"

        self.robot.setMode(self.mode.NRT_JOINT_POSITION)
        self.robot.getRobotStates(self.robot_states)
        self.gripper.move(0.07, 0.1, 5)
        # if move_home:
        #     self.move_to_home()

        self.robot.setMode(self.mode.NRT_JOINT_POSITION)

        if init_qpos is not None:
            assert len(init_qpos) == 7
            self.send_joint_position(np.array(init_qpos))
            time.sleep(12)
            print("Set to desired q pos: ", init_qpos)

        if init_offset is not None:
            assert len(init_offset) == 3
            pose = self.get_ee_pose()
            pose = Flexiv.tip_to_flange_pose(pose)
            pos, rot = pose_to_pos_rot(pose)
            pos[0] += init_offset[0]
            pos[1] += init_offset[1]
            pos[2] += init_offset[2]
            pose = pos_rot_to_pose(pos, rot)
            self.send_flange_pose(pose)
            print("Set position offset: ", init_offset)
            time.sleep(12)

        self.log.info("Done robot initializing")

        self.robot.getRobotStates(self.robot_states)
        self.sim_rizon.set_joints(self.robot_states.q)

        # print(self.mode.__dict__)
        # self.robot.setMode(self.mode.RT_JOINT_POSITION)

    def move_to_home(self):
        self.log.info("Move to home")
        # robot
        self.robot.setMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.executePrimitive("Home()")
        while self.robot.isBusy():
            time.sleep(1)
        self.robot.executePrimitive("ZeroFTSensor()")

        # gripper
        self.gripper.move(0.08, 0.1, 5)

        time.sleep(1.0)
        self.log.info("Moved home")
        print(self.get_gripper_width())

    # robot arm api

    def get_flange_pose(self):
        """return pose in flexiv's coordinates"""
        self.robot.getRobotStates(self.robot_states)


        ### ABANDON: direct read flangePose from api
        # flange_pose = self.robot_states.flangePose  # pppqqqq wxyz quat for flexiv api
        # pos, quat = flange_pose[:3], flange_pose[3:]
        # pos = np.array(pos)
        # rot = st.Rotation.from_quat(quat, scalar_first=True)

        ### read flangepose from pybullet
        self.sim_rizon.set_joints(self.get_joint_positions())
        flange_pose = self.sim_rizon.get_catersian(self.sim_rizon.flange_link)
        pos, quat = flange_pose[:3], flange_pose[3:]  # xyzw quat for pybullet
        rot = st.Rotation.from_quat(quat, scalar_first=False)

        return pos_rot_to_pose(pos, rot)

    def get_ee_pose(self):
        
        flange_pose = self.get_flange_pose()
        pos, rot = pose_to_pos_rot(flange_pose)

        tip_pose_mat = pos_rot_to_mat(np.array(pos), rot) @ Flexiv.tx_flange_tip
        umi_tip_pose = mat_to_pose(tip_pose_mat)
        # print("tip pose", umi_tip_pose)

        return umi_tip_pose

    def get_tcp(self):
        self.robot.getRobotStates(self.robot_states)
        return np.array(self.robot_states.tcpPose) # pppqqqq wxyz quat for flexiv api

    def get_joint_positions(self):
        self.robot.getRobotStates(self.robot_states)
        return np.array(self.robot_states.q)

    def get_joint_velocities(self):
        self.robot.getRobotStates(self.robot_states)
        return np.array(self.robot_states.dq)

    def readPose(self):
        self.robot.getRobotStates(self.robot_states)
        flange_pose = self.robot_states.tcpPose  # pppqqqq wxyz quat for flexiv api
        pos, quat = flange_pose[:3], flange_pose[3:]
        print("quat", quat)
        pos = np.array(pos)
        rot = st.Rotation.from_quat([quat[1],quat[2],quat[3],quat[0]])
        rot = rot.as_matrix()
        T = np.eye(4)
        T[0, 3] = pos[0]
        T[1, 3] = pos[1]
        T[2, 3] = pos[2]
        T[:3, :3] = rot
        return T
    
    def readCamPose(self):
        self.robot.getRobotStates(self.robot_states)
        cam_pose = self.robot_states.camPose
        pos, quat = cam_pose[:3], cam_pose[3:]
        print("quat", quat)
        pos = np.array(pos)
        rot = st.Rotation.from_quat([quat[1],quat[2],quat[3],quat[0]])
        rot = rot.as_matrix()
        T = np.eye(4)
        T[0, 3] = pos[0]
        T[1, 3] = pos[1]
        T[2, 3] = pos[2]
        T[:3, :3] = rot
        
        return T

    def movePose(self, pose):
        # gripper pose
        # tf.euler_from_matrix(pose, axes='rzyx')
        # R.from_euler('ZYX', [-1.560670, -0.745688, 1.922058]).as_matrix(), rpy->matrix

        self.send_flange_pose(pose)

    def movePosePrimitive(self, pose):
        # gripper pose
        # tf.euler_from_matrix(pose, axes='rzyx')
        # R.from_euler('ZYX', [-1.560670, -0.745688, 1.922058]).as_matrix(), rpy->matrix
        tr = pose[:3, 3]
        rot = tf.euler_from_matrix(pose, axes='rzyx')
        rot_deg = np.degrees(rot)
        rot_deg = np.around(rot_deg, decimals=4)
        target_point = f"{tr[0]} {tr[1]} {tr[2]} {rot_deg[2]} {rot_deg[1]} {rot_deg[0]}"
        self.robot.setMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.executePrimitive(
            "MoveL(target= "
            + target_point
            + " WORLD WORLD_ORIGIN, maxVel=0.05)"
        )
        
        while parse_pt_states(self.robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
                

    
    def send_flange_pose(self, flange_pose):
        
        print("Arm: ", time.monotonic(), flange_pose)
        """receive pose in flexiv's coordinates, not umi coordinates"""
        self.sim_rizon.set_joints(self.get_joint_positions())

        ### Protect
        #tcp_pose = mat_to_pose(pose_to_mat(flange_pose) @ Flexiv.tx_flange_tip)
        tcp_pose = flange_pose
        #tcp_pose[2] = max(tcp_pose[2], 0.03)  # limit z
        flange_pose = mat_to_pose(tcp_pose) #@ Flexiv.tx_tip_flange
        print("flange_pose",flange_pose)
        

        # print("Send flange", " ".join(["%5.2f"%x for x in flange_pose]))

        # from pos-rotvec 6d pose to pos-quat 7d pose
        pos, rot = pose_to_pos_rot(flange_pose)
        pos[2] += 0.148
        print("pos",pos)
        quat = rot.as_quat()  # zyxw quat for scipy/pybullet api
        # quat = quat / np.linalg.norm(quat)
        flange_pose = np.concatenate([pos, quat])
        next_joints = self.sim_rizon.calc_ik(flange_pose)  # pppqqqq -> q
        print("next_joints",next_joints)

        if self.verbose:
            print(
                "[Flexiv] [DEBUG] Sending flange pose:",
                " ".join(["%4.2f" % x for x in flange_pose]),
                "(q =",
                " ".join(["%4.2f" % x for x in next_joints]),
                ")",
            )
        self.send_joint_position(next_joints)

    def send_joint_position(self, positions: np.ndarray):
        
        if os.environ.get("FLEXIV_USE_VEL_CONTROL", "0") == "1":
            if self.last_send_pose is None:
                target_vel = np.zeros_like(positions)
            else:
                curr_pos = self.get_joint_positions()
                curr_t = time.time()
                target_vel = (positions - curr_pos) / (curr_t - self.last_send_pose)
                target_vel[target_vel > Flexiv.MAX_VEL] = Flexiv.MAX_VEL
                self.last_send_pose = curr_t
        else:
            target_vel = Flexiv.TARGET_VEL

        print("positions",positions)
        #exit(0)
        # for rizon4
        self.robot.sendJointPosition(
            positions,
            target_vel,
            Flexiv.TARGET_ACC,
            Flexiv.MAX_VEL,
            Flexiv.MAX_ACC,
        )
        # self.robot.streamJointPosition(
        #     positions,
        #     Flexiv.TARGET_VEL,
        #     Flexiv.TARGET_ACC,
        # )

    # gripper api

    def send_gripper_state(self, pos: float, vel: float, force: float):
        print("Gripper: ", time.monotonic(), pos)
        target_pos = pos
        # print("Send gripper", "%5.2f"%target_pos)
        if self.verbose:
            print("[Flexiv] [DEBUG] Gripper move to %.5f" % target_pos)
        self.gripper.move(target_pos, vel, force)
        time.sleep(1.0)

    def get_gripper_width(self):
        self.gripper.getGripperStates(self.gripper_states)
        return self.gripper_states.width

    def get_gripper_force(self):
        self.gripper.getGripperStates(self.gripper_states)
        return self.gripper_states.force

    def get_gripper_state(self):
        self.gripper.getGripperStates(self.gripper_states)
        return self.gripper_states.width, self.gripper_states.force

if __name__ == "__main__":

    HOME_POSE = json.load(open("/home/rhos/xiaoyang/calib_out/capture_0305/door/view_002_pose_000/robot_pose.json", "r"))
    HOME_POSE = np.array(HOME_POSE)
    robot = Flexiv()
    # robot.movePosePrimitive(HOME_POSE)
    robot.send_gripper_state(0.8 ,0.1, 20)
    print("pose",robot.readCamPose())
    time.sleep(1.0)