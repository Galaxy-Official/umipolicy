import time
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np
import pybullet as pb

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class SessionClaim:
    operator_id: str
    robot_id: str
    mode: str = "supervisory"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "claim_session"
        return payload


@dataclass
class SessionReply:
    accepted: bool
    session_id: str
    reason: str = ""
    robot_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "session_reply"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionReply":
        return cls(
            accepted=bool(payload.get("accepted", False)),
            session_id=str(payload.get("session_id", "")),
            reason=str(payload.get("reason", "")),
            robot_id=str(payload.get("robot_id", "")),
        )


@dataclass
class OperatorCommand:
    session_id: str
    operator_id: str
    command: str
    sent_at_unix_ms: int = field(default_factory=now_ms)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "operator_command"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperatorCommand":
        return cls(
            session_id=str(payload.get("session_id", "")),
            operator_id=str(payload.get("operator_id", "")),
            command=str(payload.get("command", "")).strip().lower(),
            sent_at_unix_ms=int(payload.get("sent_at_unix_ms", now_ms())),
        )


@dataclass
class CommandReply:
    accepted: bool
    command: str
    session_id: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "command_reply"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CommandReply":
        return cls(
            accepted=bool(payload.get("accepted", False)),
            command=str(payload.get("command", "")),
            session_id=str(payload.get("session_id", "")),
            reason=str(payload.get("reason", "")),
        )


@dataclass
class LeaderState:
    session_id: str
    operator_id: str
    seq: int
    sent_at_unix_ms: int
    leader_joint_pos: list[float]
    leader_gripper: float
    estop_pressed: bool = False
    source_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "leader_state"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LeaderState":
        return cls(
            session_id=str(payload.get("session_id", "")),
            operator_id=str(payload.get("operator_id", "")),
            seq=int(payload.get("seq", 0)),
            sent_at_unix_ms=int(payload.get("sent_at_unix_ms", now_ms())),
            leader_joint_pos=[float(v) for v in payload.get("leader_joint_pos", [])],
            leader_gripper=float(payload.get("leader_gripper", 0.0)),
            estop_pressed=bool(payload.get("estop_pressed", False)),
            source_ok=bool(payload.get("source_ok", True)),
        )


@dataclass
class MappedTeleopTarget:
    joint_target: list[float]
    gripper_width: float
    eef_pos: list[float]
    eef_orn: list[float]


@dataclass
class RobotTelemetry:
    session_id: str
    robot_joint_pos: list[float] = field(default_factory=list)
    robot_gripper: float = 0.0
    robot_fault: bool = False
    robot_mode: str = "unknown"
    last_cmd_age_ms: int = -1
    link_state: str = "idle"
    safety_state: str = "idle"
    status_message: str = ""
    updated_at_unix_ms: int = field(default_factory=now_ms)
    camera_urls: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "telemetry"
        return payload


class LocalLeaderDevice:
    def __init__(
        self,
        robot_path: str,
        robot_overrides: list[str] | None = None,
        arm_name: str = "main",
    ):
        self.robot_path = robot_path
        self.robot_overrides = robot_overrides or ["~follower_arms", "~cameras"]
        self.arm_name = arm_name
        self.robot = None
        self.arm = None

    def connect(self):
        robot_cfg = init_hydra_config(self.robot_path, self.robot_overrides)
        self.robot = make_robot(robot_cfg)
        self.robot.connect()
        self.arm = self.robot.leader_arms[self.arm_name]

    def read_leader_state(self, session_id: str, operator_id: str, seq: int) -> LeaderState:
        if self.arm is None:
            raise RuntimeError("Leader device is not connected.")

        leader_joint_pos = self.arm.read("Present_Position")
        leader_joint_pos = np.asarray(leader_joint_pos, dtype=float).tolist()
        leader_gripper = float(leader_joint_pos[-1]) if leader_joint_pos else 0.0
        return LeaderState(
            session_id=session_id,
            operator_id=operator_id,
            seq=seq,
            sent_at_unix_ms=now_ms(),
            leader_joint_pos=leader_joint_pos,
            leader_gripper=leader_gripper,
        )

    def disconnect(self):
        if self.robot is not None and getattr(self.robot, "is_connected", False):
            self.robot.disconnect()
        self.robot = None
        self.arm = None


class LeaderToFollowerMapper:
    URDF_DICT = {
        "koch": "urdf/assets/low_cost_robot_description/urdf/low_cost_robot.urdf",
        "so100": "urdf/assets/SO_5DOF_ARM100_8j_URDF.SLDASM/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf",
    }

    def __init__(self, leader_robot_path: str, robot_overrides: list[str] | None = None):
        self.leader_robot_path = leader_robot_path
        self.robot_overrides = robot_overrides or ["~cameras", "~follower_arms"]
        self.robot_cfg = init_hydra_config(self.leader_robot_path, self.robot_overrides)
        self.robot_type = self.robot_cfg["robot_type"]
        self.physics_client = pb.connect(pb.DIRECT)

        urdf_path = self.URDF_DICT.get(self.robot_type)
        if urdf_path is None:
            raise ValueError(f"Unsupported leader robot type: {self.robot_type}")

        self.robot_pb = pb.loadURDF(
            urdf_path,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0, 0, 1, 0] if self.robot_type == "so100" else [0, 0, 0, 1],
            useFixedBase=True,
        )
        self.follow_arm = pb.loadURDF(
            "urdf/assets/leap_hand/flexiv_leap.urdf",
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0, 0, 0.7071068, 0.7071068],
            useFixedBase=True,
        )

    def leader_to_eef(self, leader_joint_pos: list[float]) -> tuple[np.ndarray, np.ndarray, float]:
        data = [float(v) for v in leader_joint_pos]

        if self.robot_type == "koch":
            data = [
                -data[0],
                90 - data[1],
                90 - data[2],
                90 - data[3],
                data[4] - 90,
                -data[5],
            ]
            data = [angle * np.pi / 180.0 for angle in data]

            for i, joint in enumerate(data):
                pb.resetJointState(self.robot_pb, i, float(joint))

            eef_pos = np.array(pb.getLinkState(self.robot_pb, 4)[0])
            eef_orn = np.array(pb.getLinkState(self.robot_pb, 4)[1])
            return eef_pos, eef_orn, -float(data[-1])

        if self.robot_type == "so100":
            data = [
                -data[0],
                90 - data[1],
                data[2] - 90,
                data[3] - 90,
                90 - data[4],
                data[5],
            ]
            data = [angle * np.pi / 180.0 for angle in data]

            for i, joint in enumerate(data):
                pb.resetJointState(self.robot_pb, i, float(joint))

            eef_pos = np.array(pb.getLinkState(self.robot_pb, 4)[0])
            eef_orn = np.array(pb.getLinkState(self.robot_pb, 4)[1])
            eef_orn_matrix = np.array(pb.getMatrixFromQuaternion(eef_orn)).reshape(3, 3)
            original_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            z_in_eef = np.array([-1, 0, 0])
            x_in_eef = np.array([0, -1, 0])
            y_in_eef = np.array([0, 0, 1])
            new_axes = np.array([x_in_eef, y_in_eef, z_in_eef]).T
            new_axes = np.concatenate([new_axes, np.array([[1, 1, 1]])], axis=0)

            x_ab = np.array(
                [
                    np.dot(eef_orn_matrix[:, 0], original_axes[0]),
                    np.dot(eef_orn_matrix[:, 0], original_axes[1]),
                    np.dot(eef_orn_matrix[:, 0], original_axes[2]),
                ]
            )
            y_ab = np.array(
                [
                    np.dot(eef_orn_matrix[:, 1], original_axes[0]),
                    np.dot(eef_orn_matrix[:, 1], original_axes[1]),
                    np.dot(eef_orn_matrix[:, 1], original_axes[2]),
                ]
            )
            z_ab = np.array(
                [
                    np.dot(eef_orn_matrix[:, 2], original_axes[0]),
                    np.dot(eef_orn_matrix[:, 2], original_axes[1]),
                    np.dot(eef_orn_matrix[:, 2], original_axes[2]),
                ]
            )
            t_ab = np.array([x_ab, y_ab, z_ab, eef_pos]).T
            t_ab = np.concatenate([t_ab, np.array([[0, 0, 0, 1]])], axis=0)
            rot_axes = (t_ab @ new_axes)[:3]
            new_eef_orn = self.matrix2quaternion(rot_axes)
            return eef_pos, new_eef_orn, -float(data[-1])

        raise ValueError(f"Unsupported leader robot type: {self.robot_type}")

    def map_leader_state(self, leader_state: LeaderState) -> MappedTeleopTarget:
        eef_pos, eef_orn, gripper = self.leader_to_eef(leader_state.leader_joint_pos)
        scaled_pos = eef_pos * 4.2
        joints = pb.calculateInverseKinematics(self.follow_arm, 8, scaled_pos, eef_orn)
        for i, joint in enumerate(joints):
            pb.resetJointState(self.follow_arm, i, joint)

        return MappedTeleopTarget(
            joint_target=[float(joint) for joint in joints],
            gripper_width=float(gripper),
            eef_pos=[float(v) for v in eef_pos.tolist()],
            eef_orn=[float(v) for v in np.asarray(eef_orn).tolist()],
        )

    def close(self):
        if self.physics_client is not None:
            pb.disconnect(self.physics_client)
            self.physics_client = None

    @staticmethod
    def matrix2quaternion(matrix):
        tr = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * s
            qx = (matrix[2, 1] - matrix[1, 2]) / s
            qy = (matrix[0, 2] - matrix[2, 0]) / s
            qz = (matrix[1, 0] - matrix[0, 1]) / s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s
        return np.array([qx, qy, qz, qw])


def encode_jpeg(frame, jpeg_quality: int = 75) -> bytes:
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image frame, got {frame.shape}.")

    rgb_frame = frame.astype("uint8", copy=False)
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG frame.")
    return encoded.tobytes()
