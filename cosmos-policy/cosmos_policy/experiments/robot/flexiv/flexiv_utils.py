import numpy as np
from scipy.spatial.transform import Rotation


def _normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, eps)


def rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def pose6_to_mat(pose6: np.ndarray) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float32)
    mat = np.zeros(pose6.shape[:-1] + (4, 4), dtype=np.float32)
    mat[..., :3, 3] = pose6[..., :3]
    mat[..., :3, :3] = Rotation.from_rotvec(pose6[..., 3:6]).as_matrix()
    mat[..., 3, 3] = 1.0
    return mat


def mat_to_pose6(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    pos = mat[..., :3, 3]
    rotvec = Rotation.from_matrix(mat[..., :3, :3]).as_rotvec().astype(np.float32)
    return np.concatenate([pos, rotvec], axis=-1).astype(np.float32)


def pose10d_to_mat(pose10d: np.ndarray) -> np.ndarray:
    pose10d = np.asarray(pose10d, dtype=np.float32)
    mat = np.zeros(pose10d.shape[:-1] + (4, 4), dtype=np.float32)
    mat[..., :3, 3] = pose10d[..., :3]
    mat[..., :3, :3] = rot6d_to_mat(pose10d[..., 3:9])
    mat[..., 3, 3] = 1.0
    return mat


def mat_to_pose10d(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    pos = mat[..., :3, 3]
    rot6d = mat[..., :2, :3].copy().reshape(mat.shape[:-2] + (6,))
    return np.concatenate([pos, rot6d], axis=-1).astype(np.float32)


def absolute_state_to_relative_pose10d(state: np.ndarray, base_state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    base_state = np.asarray(base_state, dtype=np.float32)
    state_mat = pose6_to_mat(state[..., :6])
    base_mat = pose6_to_mat(base_state[:6])
    rel_mat = np.linalg.inv(base_mat) @ state_mat
    gripper = state[..., 6:7]
    return np.concatenate([mat_to_pose10d(rel_mat), gripper], axis=-1).astype(np.float32)


def relative_pose10d_actions_to_absolute(actions: np.ndarray, base_state: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    base_state = np.asarray(base_state, dtype=np.float32)
    if actions.shape[-1] != 10:
        raise ValueError(f"Expected Flexiv relative pose10d actions with dim 10, got shape {actions.shape}")

    rel_mat = pose10d_to_mat(actions[..., :9])
    base_mat = pose6_to_mat(base_state[:6])
    abs_mat = base_mat @ rel_mat
    pose6 = mat_to_pose6(abs_mat)
    gripper = actions[..., 9:10]
    return np.concatenate([pose6, gripper], axis=-1).astype(np.float32)
