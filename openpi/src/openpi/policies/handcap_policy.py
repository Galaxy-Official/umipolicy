import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/head_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/side_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_tactile": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_tactile": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/force": np.random.uniform(0.0, 10.0, size=(2,)).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _force_from_value(value, force_dim: int) -> np.ndarray:
    force = np.asarray(value, dtype=np.float32)
    if force.ndim == 0:
        force = force[None]
    force = np.reshape(force, (-1,))
    if force.shape[-1] < force_dim:
        force = np.pad(force, (0, force_dim - force.shape[-1]))
    return force[:force_dim]


def _extract_force(data: dict, state: np.ndarray, force_dim: int, action_base_dim: int) -> np.ndarray:
    for key in (
        "observation/force",
        "observation/forces",
        "observation.force",
        "observation.forces",
        "force",
    ):
        if key in data:
            return _force_from_value(data[key], force_dim)

    left = None
    right = None
    for key in ("observation/force_left", "observation.forces.left", "observation/forces/left"):
        if key in data:
            left = _force_from_value(data[key], 1)
            break
    for key in ("observation/force_right", "observation.forces.right", "observation/forces/right"):
        if key in data:
            right = _force_from_value(data[key], 1)
            break
    if left is not None and right is not None:
        return _force_from_value(np.concatenate([left, right], axis=-1), force_dim)

    state = np.asarray(state)
    if state.shape[-1] >= action_base_dim + force_dim:
        return _force_from_value(state[..., action_base_dim : action_base_dim + force_dim], force_dim)

    return np.zeros((force_dim,), dtype=np.float32)


def _append_force_to_state(state: np.ndarray, force: np.ndarray, action_base_dim: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] >= action_base_dim + force.shape[-1]:
        return state
    return np.concatenate([state[..., :action_base_dim], force.astype(state.dtype)], axis=-1)


def _ensure_action_force(actions: np.ndarray, force_dim: int, action_base_dim: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    output_dim = action_base_dim + force_dim
    if actions.shape[-1] >= output_dim:
        return actions[..., :output_dim]
    if actions.shape[-1] < action_base_dim:
        actions = transforms.pad_to_dim(actions, action_base_dim, axis=-1)
    force_pad_shape = actions.shape[:-1] + (force_dim,)
    force_pad = np.zeros(force_pad_shape, dtype=actions.dtype)
    return np.concatenate([actions[..., :action_base_dim], force_pad], axis=-1)

def _pointcloud_from_phone_rgbd(
    depth: np.ndarray,
    rgb: np.ndarray | None,
    num_points: int,
    point_dim: int,
) -> np.ndarray:
    """
    Minimal RGB-D to pointcloud fallback for smoke test.

    Assumption:
        depth shape: [H, W, 1] or [H, W]
        rgb shape: [H, W, 3] or [3, H, W], optional

    Output:
        [num_points, point_dim]
    """
    depth = np.asarray(depth)

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]

    depth = depth.astype(np.float32)

    # 如果 depth 是 uint16/mm，转成 meter；如果本来就是 meter，这一步不会太离谱但可按实际再修
    if depth.max() > 20.0:
        depth = depth / 1000.0

    H, W = depth.shape

    ys, xs = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )

    # smoke test 用的简化内参。正式训练应换成真实 phone intrinsic。
    fx = fy = float(max(H, W))
    cx = float(W - 1) / 2.0
    cy = float(H - 1) / 2.0

    z = depth
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z

    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    valid = np.isfinite(xyz).all(axis=-1) & (xyz[:, 2] > 1e-6)
    xyz = xyz[valid]

    if point_dim <= 3:
        pc = xyz
    else:
        if rgb is None:
            color = np.zeros((xyz.shape[0], point_dim - 3), dtype=np.float32)
        else:
            rgb_arr = _parse_image(rgb)
            if rgb_arr.shape[:2] != (H, W):
                # smoke test 简化处理：不做 resize，直接不用颜色
                color = np.zeros((xyz.shape[0], point_dim - 3), dtype=np.float32)
            else:
                rgb_flat = rgb_arr.reshape(-1, 3)[valid].astype(np.float32) / 255.0
                if point_dim - 3 <= 3:
                    color = rgb_flat[:, : point_dim - 3]
                else:
                    extra = np.zeros((rgb_flat.shape[0], point_dim - 6), dtype=np.float32)
                    color = np.concatenate([rgb_flat, extra], axis=-1)
        pc = np.concatenate([xyz, color], axis=-1)

    return _sample_or_pad_pointcloud(pc, num_points, point_dim)

def _sample_or_pad_pointcloud(
    pointcloud: np.ndarray,
    num_points: int,
    point_dim: int,
    *,
    random_sample: bool = True,
) -> np.ndarray:
    """
    Convert an arbitrary pointcloud to fixed shape [num_points, point_dim].

    Args:
        pointcloud:
            Supported shapes:
              - [N, C]
              - [H, W, C]
              - [1, N, C]  # will squeeze leading singleton batch
            C can be >= point_dim. Extra channels are truncated.
        num_points:
            Target number of points.
        point_dim:
            Target point feature dimension, e.g. 3 for xyz, 6 for xyzrgb.
        random_sample:
            If True, randomly sample when N > num_points.
            If False, use deterministic evenly-spaced sampling.

    Returns:
        pc_out: np.float32 array with shape [num_points, point_dim].
    """
    pc = np.asarray(pointcloud, dtype=np.float32)

    # Remove a singleton batch dimension if present: [1, N, C] -> [N, C]
    if pc.ndim == 3 and pc.shape[0] == 1 and pc.shape[-1] <= 16:
        pc = pc[0]

    # Flatten image-like pointcloud: [H, W, C] -> [H*W, C]
    if pc.ndim == 3:
        pc = pc.reshape(-1, pc.shape[-1])

    if pc.ndim != 2:
        raise ValueError(f"pointcloud must be [N,C] or [H,W,C], got shape={pc.shape}")

    if pc.shape[-1] < 3:
        raise ValueError(f"pointcloud last dim must be at least xyz=3, got shape={pc.shape}")

    # Remove invalid rows. This is important for online RGBD back-projection.
    finite_mask = np.isfinite(pc).all(axis=-1)
    pc = pc[finite_mask]

    # Optional: remove all-zero xyz rows, common when depth is invalid.
    if pc.shape[0] > 0:
        nonzero_xyz = np.linalg.norm(pc[:, :3], axis=-1) > 1e-8
        pc = pc[nonzero_xyz]

    # Truncate or pad feature dim.
    if pc.shape[-1] >= point_dim:
        pc = pc[:, :point_dim]
    else:
        pad = np.zeros((pc.shape[0], point_dim - pc.shape[-1]), dtype=np.float32)
        pc = np.concatenate([pc, pad], axis=-1)

    # If no valid points, return all-zero padded cloud.
    if pc.shape[0] == 0:
        return np.zeros((num_points, point_dim), dtype=np.float32)

    n = pc.shape[0]

    if n >= num_points:
        if random_sample:
            idx = np.random.choice(n, size=num_points, replace=False)
        else:
            idx = np.linspace(0, n - 1, num_points).astype(np.int64)
        pc = pc[idx]
    else:
        # Pad by repeating valid points, then truncate to exact length.
        repeat_idx = np.random.choice(n, size=num_points - n, replace=True)
        pc = np.concatenate([pc, pc[repeat_idx]], axis=0)

    return pc.astype(np.float32, copy=False)




@dataclasses.dataclass(frozen=True)
class HandcapInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0
    include_tactile: bool = True
    force_predict: bool = False
    force_guide: bool = False
    action_base_dim: int = 10
    force_dim: int = 2
    
    use_point: bool = False
    point_num_points: int = 1024
    point_dim: int = 3
    point_key: str = "phone_pointcloud"

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        raw_state = np.asarray(data["observation/state"], dtype=np.float32)
        force = _extract_force(data, raw_state, self.force_dim, self.action_base_dim)
        if self.force_predict or self.force_guide:
            raw_state = _append_force_to_state(raw_state, force, self.action_base_dim)
        state = transforms.pad_to_dim(raw_state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        wrist_image = _parse_image(data["observation/wrist_image"])
        
        if self.include_tactile and "observation/left_tactile" in data:
            left_tactile = _parse_image(data["observation/left_tactile"])
            right_tactile = _parse_image(data["observation/right_tactile"])
            tactile_mask = np.True_
        else:
            left_tactile = np.zeros_like(wrist_image)
            right_tactile = np.zeros_like(wrist_image)
            tactile_mask = np.False_

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "force": force,
            "image": {
                'wrist_0_rgb': wrist_image, 
                'left_tactile_0_rgb': left_tactile,
                'right_tactile_0_rgb': right_tactile
            },
            "image_mask": {
                "wrist_0_rgb": np.True_,
                "left_tactile_0_rgb": tactile_mask,
                "right_tactile_0_rgb": tactile_mask
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            
            actions = np.asarray(data["actions"], dtype=np.float32)
            if self.force_predict:
                actions = _ensure_action_force(actions, self.force_dim, self.action_base_dim)
            actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
        if self.use_point and "observation/pointcloud" in data:
            pc = np.asarray(data["observation/pointcloud"], dtype=np.float32)
            pc = _sample_or_pad_pointcloud(pc, self.point_num_points, self.point_dim)
            inputs["pointcloud"] = {self.point_key: pc}
            inputs["pointcloud_mask"] = {self.point_key: np.True_}

        elif self.use_point and "observation/phone_depth" in data:
            depth = np.asarray(data["observation/phone_depth"])
            rgb = data.get("observation/phone_image", None)
            pc = _pointcloud_from_phone_rgbd(
                depth=depth,
                rgb=rgb,
                num_points=self.point_num_points,
                point_dim=self.point_dim,
            )
            inputs["pointcloud"] = {self.point_key: pc}
            inputs["pointcloud_mask"] = {self.point_key: np.True_}

        elif self.use_point:
            inputs["pointcloud"] = {
                self.point_key: np.zeros((self.point_num_points, self.point_dim), dtype=np.float32),
            }
            inputs["pointcloud_mask"] = {
                self.point_key: np.False_,
            }

        return inputs


@dataclasses.dataclass(frozen=True)
class HandcapOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    output_action_dim: int = 10
    force_predict: bool = False
    action_base_dim: int = 10
    force_dim: int = 2

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        actions = np.asarray(data["actions"][:, : self.output_action_dim])
        outputs = {"actions": actions}
        if self.force_predict:
            outputs["robot_actions"] = actions[:, : self.action_base_dim]
            outputs["gripper_pred"] = actions[:, self.action_base_dim - 1 : self.action_base_dim]
            outputs["force_pred"] = actions[:, self.action_base_dim : self.action_base_dim + self.force_dim]
        return outputs
