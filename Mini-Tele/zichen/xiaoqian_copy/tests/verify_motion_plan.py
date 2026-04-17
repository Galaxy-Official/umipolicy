import os
import time
import numpy as np
import pybullet as p
from typing import List, Tuple, Optional
import pathlib
import random
from collections import deque
import sys
import os.path as osp

# Add pybullet_ompl to path
# try:
sys.path.insert(0, '/Disk3/xiaoqian/pybullet_ompl')
import pb_ompl
    # Test if OMPL is actually available by trying to import the core modules
    # try:
    #     from ompl import base as ob
    #     from ompl import geometric as og
    #     from ompl import util as ou
    #     print("✓ pybullet_ompl and OMPL are available")
    # except ImportError as ompl_error:
    #     print(f"⚠ Warning: pybullet_ompl found but OMPL C++ bindings not available: {ompl_error}")
    #     print("⚠ Falling back to custom RRT implementation")
    #     pb_ompl = None
# except ImportError as pb_error:
#     print(f"⚠ Warning: pybullet_ompl not found: {pb_error}")
#     print("⚠ Falling back to custom RRT implementation")
#     pb_ompl = None

class Rizon4:
    eps: float = 1.0e-5
    
    def __init__(self, headless: bool=True):
        self.headless = headless
        p.connect(p.DIRECT if self.headless else p.GUI)

        # self.urdf_file = os.path.join(os.path.dirname(__file__), 'assets/flexiv_rizon4.urdf')
        
        # script_dir = pathlib.Path(__file__).parent.parent.parent.joinpath('/home/rhos/xiaoyang/flexiv_api/assets/flexiv_rizon4.urdf')
        script_dir = pathlib.Path(__file__).parent.parent.parent.joinpath('/Disk3/xiaoqian/flexiv_api/assets/flexiv_rizon4.urdf')

        self.urdf_file = str(script_dir)
        print("Load from URDF:", self.urdf_file)
        self.robot: int = p.loadURDF(self.urdf_file)    # body index
        self.physics_id = 0  # Default physics client ID for primary connection

        self.num_joints = 7
        self.flange_link = 7
        self.tcp_link = 10                              # closed_fingers_tcp: 10
        self.joint_indices = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
        
        print("Total joint number: ", p.getNumJoints(self.robot))
        # num_joints = p.getNumJoints(self.robot)
        # for i in range(num_joints):
        #     ret = p.getJointInfo(self.robot, i)
        #     print(i, ret[1], ret[-5], ret)
    
    @staticmethod
    def clear_debug():
        p.removeAllUserDebugItems()

        
    def visualize(self, link_id: int=-1):
        if link_id == -1:
            link_id = self.tcp_link
        state = p.getLinkState(self.robot, link_id)
        position = state[0]
        orientation = state[1]
        self.visualize_coords(position, orientation)
    
    def visualize_point(self, position: np.ndarray, color: Tuple[int]=(0, 0, 0), size: float=5):
        p.addUserDebugPoints([position], [color], size)
        
    def visualize_coords(self, position: np.ndarray, orientation: np.ndarray):
        mat = np.eye(4)
        mat[:3, 3] = position
        mat[:3, :3] = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        p.addUserDebugLine(mat[:3,3], mat[:3,3] + mat[:3,0]*0.3, (1,0,0), 3.0)
        p.addUserDebugLine(mat[:3,3], mat[:3,3] + mat[:3,1]*0.3, (0,1,0), 3.0)
        p.addUserDebugLine(mat[:3,3], mat[:3,3] + mat[:3,2]*0.3, (0,0,1), 3.0)        
    
    def get_dof_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """dof limits (8-dofs, including gripper)

        Returns:
            Tuple[List[float], List[float]]: _description_
        """
        p.getNumJoints(self.robot)

        dof_lower_limits = []
        dof_upper_limits = []
            
        for i in self.joint_indices:
            ret = p.getJointInfo(self.robot, i)    
            dof_lower_limits.append(ret[8])
            dof_upper_limits.append(ret[9])
        
        # gripper width limits
        # dof_lower_limits.append(0)
        # dof_upper_limits.append(0.10)
        return np.array(dof_lower_limits, dtype=np.float32), np.array(dof_upper_limits, dtype=np.float32)
    
    @staticmethod
    def get_tcp_limits() -> Tuple[np.ndarray, np.ndarray]:
        """tcp limits

        Returns:
            Tuple[List[float], List[float]]: _description_
        """
        tcp_lower_bound = np.array([0.50, -0.24, 0.20, -1, -1, -1, -1], dtype=np.float32) # xyz, wxyz, griv
        tcp_upper_bound = np.array([0.82,  0.32, 0.55,  1,  1,  1,  1], dtype=np.float32)
        
        return tcp_lower_bound, tcp_upper_bound
    
    @staticmethod
    def get_flange_limits() -> Tuple[np.ndarray, np.ndarray]:
        """tcp limits

        Returns:
            Tuple[List[float], List[float]]: _description_
        """
        flange_lower_bound = np.array([0.45, -0.24, 0.23, -np.pi, -np.pi, -np.pi], dtype=np.float32) # xyz, wxyz, griv
        flange_upper_bound = np.array([0.82,  0.32, 0.55,  np.pi,  np.pi,  np.pi], dtype=np.float32)
        
        return flange_lower_bound, flange_upper_bound

    def get_catersian(self, link_id: int):
        """Get cartesian states

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        p.stepSimulation()
        state = p.getLinkState(self.robot, link_id)
        position = state[0]
        orientation = state[1]
        return np.array(position + orientation)
    
    def get_flange_catersian(self):
        """get flange cartesian states"""
        return self.get_catersian(link_id=self.flange_link)
    
    def get_tcp_catersian(self):
        """get tcp cartesian states"""
        return self.get_catersian(link_id=self.tcp_link)
    
    def get_joints(self):
        p.stepSimulation()
        return [p.getJointState(self.robot, i)[0] for i in self.joint_indices]
    
    def set_joints(self, joint_states: np.ndarray):
        """Set joint states

        Args:
            joint_states (np.ndarray): _description_
        """
        for i, joint_state in enumerate(joint_states):
            idx = self.joint_indices[i]
            assert p.getJointInfo(self.robot, idx)[2] == 0
            p.resetJointState(self.robot, idx, joint_state)
        # print("@@@@@@@@")
        # print(joint_states)
        # # get joint state
        # for i in self.joint_indices:
        #     print(p.getJointState(self.robot, i)[0])
        # print('########')
        p.stepSimulation()
    
    def calc_joints(self, link_id: int, position: np.ndarray, orientation: np.ndarray) -> List[float]:
        """Get joint states

        Args:
            position (np.ndarray): _description_
            orientation (np.ndarray): w, x, y, z

        Returns:
            np.ndarray: _description_
        """
        joint_states = p.calculateInverseKinematics(
            bodyIndex=self.robot,
            endEffectorLinkIndex=link_id,
            targetPosition=position,
            targetOrientation=orientation
        )
        # print(self.joint_indices, self.joint_indices.dtype, len(joint_states))
        joint_states = [joint_states[i] for i in self.joint_indices]
        return joint_states
    
    def calc_ik(self, tq: np.ndarray, is_tcp: bool=False):
        """tq represents postion, orientation (w, x, y, z)

        Args:
            position (np.ndarray): _description_
            orientation (np.ndarray): _description_
        """
        position = tq[:3]
        orientation = tq[3:]

        # print("sim_pos",self.get_catersian(7))
        # print("real_position",position)
        
        p.stepSimulation()
        link_id = self.tcp_link if is_tcp else self.flange_link
        for t in range(1000):
            joints = self.calc_joints(link_id=link_id, position=position, orientation=orientation)
            self.set_joints(joints)
            p.stepSimulation()
            delta = np.linalg.norm(tq - self.get_catersian(link_id))
            if delta < Rizon4.eps:
                break
        assert t < 1000, 'calc_ik failed!'
        return np.array(self.get_joints())

    def check_collision(self, joint_states: np.ndarray) -> bool:
        """Check if joint configuration is in collision

        Args:
            joint_states (np.ndarray): Joint angles to check

        Returns:
            bool: True if in collision, False otherwise
        """
        # Save current state
        current_joints = self.get_joints()

        # Set test joints
        self.set_joints(joint_states)
        p.stepSimulation()

        # Check for collisions
        # Get all contact points involving the robot
        contact_points = p.getContactPoints(bodyA=self.robot)

        # Filter out self-collisions (contacts between different links of the same robot)
        # and contacts with the ground plane (if present)
        external_collisions = []
        for contact in contact_points:
            # contact structure: [contactFlag, bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB, ...]
            body_a = contact[1]
            body_b = contact[2]
            link_a = contact[3]
            link_b = contact[4]

            # Skip self-collisions (same body, different links)
            if body_a == body_b and link_a != link_b:
                continue

            # Skip if it's the same link (shouldn't happen but just in case)
            if body_a == body_b and link_a == link_b:
                continue

            # Skip ground contact (assuming ground is body_id 0 or plane)
            if body_b == 0:  # Ground plane
                continue

            external_collisions.append(contact)

        # Restore original state
        self.set_joints(current_joints)

        # Return True if any external collision detected
        return len(external_collisions) > 0

    def interpolate_joints(self, start_joints: np.ndarray, end_joints: np.ndarray,
                          num_waypoints: int = 50) -> List[np.ndarray]:
        """Interpolate between two joint configurations

        Args:
            start_joints (np.ndarray): Starting joint configuration
            end_joints (np.ndarray): Target joint configuration
            num_waypoints (int): Number of intermediate waypoints

        Returns:
            List[np.ndarray]: List of joint configurations along path
        """
        path = []
        start_joints = np.array(start_joints)
        end_joints = np.array(end_joints)

        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            interpolated = start_joints * (1 - t) + end_joints * t
            path.append(interpolated)
        return path

    def validate_path(self, path: List[np.ndarray]) -> Tuple[bool, int]:
        """Validate a joint path for collisions

        Args:
            path (List[np.ndarray]): List of joint configurations

        Returns:
            Tuple[bool, int]: (is_valid, collision_index) - collision_index is -1 if valid
        """
        for i, joint_states in enumerate(path):
            if self.check_collision(joint_states):
                return False, i
        return True, -1


class Rizon4PbOMPL:
    """Wrapper for Rizon4 robot to work with pybullet_ompl"""

    def __init__(self, rizon4_robot: Rizon4):
        """Initialize the wrapper with existing Rizon4 robot

        Args:
            rizon4_robot (Rizon4): The Rizon4 robot instance
        """
        self.rizon4 = rizon4_robot
        self.id = rizon4_robot.robot  # PyBullet body ID
        # Use existing connection instead of creating a new one
        # This ensures we use the same GUI/DIRECT mode as the main robot
        self.physics_id = rizon4_robot.physics_id if hasattr(rizon4_robot, 'physics_id') else p.connect(p.DIRECT)

        # Set up joint indices (same as Rizon4)
        self.joint_idx = list(rizon4_robot.joint_indices)
        self.num_dim = len(self.joint_idx)

        # Get joint bounds from Rizon4
        self.joint_bounds = []
        self.get_joint_bounds()

        # Current state
        self.state = self.get_cur_state()

        # Add env attribute for pybullet_ompl compatibility
        self.env = type('Env', (), {'id': self.physics_id})()

    def get_joint_bounds(self):
        """Get joint bounds from Rizon4 robot"""
        lower_limits, upper_limits = self.rizon4.get_dof_limits()

        for i in range(self.num_dim):
            low = float(lower_limits[i])  # Convert to Python float
            high = float(upper_limits[i])  # Convert to Python float

            # Add small margin to avoid boundary issues
            if low < high:
                delta = 0.05 * (high - low)
                low += delta
                high -= delta
                self.joint_bounds.append([low, high])

        return self.joint_bounds

    def get_cur_state(self):
        """Get current joint state"""
        return self.rizon4.get_joints()

    def set_state(self, state):
        """Set robot state for collision checking"""
        self.rizon4.set_joints(state)
        self.state = state

    def reset(self):
        """Reset robot to zero state"""
        zero_state = [0.0] * self.num_dim
        self.set_state(zero_state)


class OMPLPlanner:
    """OMPL-based path planner using pybullet_ompl"""

    def __init__(self, robot: Rizon4, planner_name: str = "RRTConnect",
                 planning_time: float = 5.0, interpolation_num: int = 100):
        """Initialize OMPL planner

        Args:
            robot (Rizon4): Rizon4 robot instance
            planner_name (str): OMPL planner to use (RRTConnect, RRTstar, BITstar, etc.)
            planning_time (float): Maximum planning time in seconds
            interpolation_num (int): Number of waypoints to interpolate
        """
        if pb_ompl is None:
            raise RuntimeError("pybullet_ompl is not available")

        self.robot = robot
        self.rizon4_pbompl = Rizon4PbOMPL(robot)
        self.obstacles = []  # Will be populated as needed

        # Create OMPL interface
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.rizon4_pbompl, self.obstacles)
        self.pb_ompl_interface.set_planner(planner_name)

        self.planning_time = planning_time
        self.interpolation_num = interpolation_num

    def add_obstacle(self, obstacle_id: int):
        """Add an obstacle to the planning environment"""
        if obstacle_id not in self.obstacles:
            self.obstacles.append(obstacle_id)
            self.pb_ompl_interface.set_obstacles(self.obstacles)

    def remove_obstacle(self, obstacle_id: int):
        """Remove an obstacle from the planning environment"""
        if obstacle_id in self.obstacles:
            self.obstacles.remove(obstacle_id)
            self.pb_ompl_interface.set_obstacles(self.obstacles)

    def plan_joint_path(self, start_joints: np.ndarray, goal_joints: np.ndarray,
                       allowed_time: float = None) -> Tuple[bool, List[np.ndarray]]:
        """Plan a path between joint configurations

        Args:
            start_joints (np.ndarray): Starting joint configuration
            goal_joints (np.ndarray): Goal joint configuration
            allowed_time (float): Planning time override

        Returns:
            Tuple[bool, List[np.ndarray]]: (success, path)
        """
        if allowed_time is None:
            allowed_time = self.planning_time

        # Convert to lists for OMPL
        start_list = start_joints.tolist()
        goal_list = goal_joints.tolist()

        # Plan path
        success, path = self.pb_ompl_interface.plan_start_goal(
            start_list, goal_list,
            allowed_time=allowed_time,
            smooth_path=True
        )

        if success and path:
            # Convert path back to numpy arrays
            path_arrays = [np.array(joint_config) for joint_config in path]
            return True, path_arrays
        else:
            return False, []

    def plan_pose_path(self, start_pose: np.ndarray, goal_pose: np.ndarray,
                      allowed_time: float = None) -> Tuple[bool, List[np.ndarray], str]:
        """Plan a path between Cartesian poses

        Args:
            start_pose (np.ndarray): Starting pose [x, y, z, qw, qx, qy, qz]
            goal_pose (np.ndarray): Goal pose [x, y, z, qw, qx, qy, qz]
            allowed_time (float): Planning time override

        Returns:
            Tuple[bool, List[np.ndarray], str]: (success, path, message)
        """
        try:
            # Calculate IK for start and goal poses
            start_joints = np.array(self.robot.calc_ik(start_pose))
            goal_joints = np.array(self.robot.calc_ik(goal_pose))
            print("start_joints", start_joints, "goal_joints", goal_joints)

            # Plan joint path
            success, path = self.plan_joint_path(start_joints, goal_joints, allowed_time)

            if success:
                message = f"OMPL planning successful - {len(path)} waypoints"
            else:
                message = "OMPL planning failed - no valid path found"

            return success, path, message

        except Exception as e:
            return False, [], f"OMPL planning error: {str(e)}"

    def visualize_path(self, path: List[np.ndarray], color: Tuple[int, int, int] = (1, 0, 0)):
        """Visualize a planned path"""
        if not path:
            return

        # Clear previous debug items
        p.removeAllUserDebugItems()

        print(f"Visualizing path with {len(path)} waypoints...")

        for i in range(len(path) - 1):
            # Save current state
            current_joints = self.robot.get_joints()

            # Set waypoint joints
            self.robot.set_joints(path[i])
            p.stepSimulation()

            # Get TCP position
            tcp_state = self.robot.get_tcp_catersian()
            pos1 = tcp_state[:3]

            # Set next waypoint
            self.robot.set_joints(path[i + 1])
            p.stepSimulation()

            # Get next TCP position
            tcp_state = self.robot.get_tcp_catersian()
            pos2 = tcp_state[:3]

            # Draw line between waypoints
            p.addUserDebugLine(pos1, pos2, color, 3.0)

            # Add waypoint markers
            if i % 5 == 0:  # Add marker every 5 waypoints to avoid clutter
                p.addUserDebugPoints([pos1], [color], pointSize=6)
                p.addUserDebugText(str(i), pos1, textColorRGB=color, textSize=1.2)

            # Restore original state
            self.robot.set_joints(current_joints)

        print("Path visualization complete")

    def execute_path(self, path: List[np.ndarray], step_duration: float = 0.05):
        """Execute a planned path

        Args:
            path: List of joint configurations
            step_duration: Time to wait between steps
        """
        if not path:
            print("No path to execute")
            return

        print(f"Executing path with {len(path)} waypoints...")

        for i, joint_states in enumerate(path):
            self.robot.set_joints(joint_states)
            p.stepSimulation()
            self.robot.visualize()

            if step_duration > 0:
                time.sleep(step_duration)

            if i % 10 == 0:
                print(f"Progress: {i+1}/{len(path)} ({(i+1)/len(path)*100:.1f}%)")

        print("Path execution complete")


if __name__ == '__main__':
    robot = Rizon4(headless=False)

    print("=== Path Planning Demo - Rizon4 Robot ===")

    print("\n开始路径规划演示！")

    # Get joint limits for reference
    low, up = robot.get_dof_limits()
    print(f"Joint limits - Lower: {low}, Upper: {up}")

    flange_pose_waypoints = np.load("pred_keypoints/capture_0301/box/view_000_pose_000/oracle_oracle/flange_pose_waypoints/000.npy")
    start_pose = flange_pose_waypoints[0]
    end_pose = flange_pose_waypoints[-1]

    # Visualize start and end positions
    robot.visualize_point(start_pose[:3], (0, 1, 0), 8)  # Green for start
    robot.visualize_point(end_pose[:3], (1, 0, 0), 8)    # Red for end

    print(f"Start pose: {start_pose}")
    print(f"End pose: {end_pose}")

    # Test both planners if OMPL is available
    # if pb_ompl is not None:
    print("\n=== Testing OMPL Planner ===")
        # try:
    ompl_planner = OMPLPlanner(robot, planner_name="RRTstar", planning_time=10.0)  # RRTConnect
    success, ompl_path, message = ompl_planner.plan_pose_path(start_pose, end_pose)
    # success, ompl_path = ompl_planner.plan_joint_path(start_pose, end_pose)
    message = "Success" if success else "Failed"
    print("ompl_path", ompl_path)
    print(f"OMPL Planning result: {message}")

    if success and ompl_path:
        print(f"OMPL path planning successful! Found path with {len(ompl_path)} waypoints")
        # ompl_planner.visualize_path(ompl_path, color=(0, 1, 1))  # Cyan for OMPL

        # Execute OMPL path
        print("\n路径执行即将开始...")
        try:
            ompl_planner.execute_path(ompl_path, step_duration=0.05)
        except KeyboardInterrupt:
            print("\n路径执行被用户中断")
    else:
        print("OMPL path planning failed!")
    

    print("\nDemo complete! Press Ctrl+C to exit.")

    while True:
        p.stepSimulation()