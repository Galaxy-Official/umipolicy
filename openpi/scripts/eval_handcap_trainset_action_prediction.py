#!/usr/bin/env python3
"""Evaluate a HandCap checkpoint on training-set samples and plot action errors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
OPENPI_SRC_ROOT = SRC_ROOT / "openpi"
CLIENT_SRC_ROOT = REPO_ROOT / "packages" / "openpi-client" / "src"
for path in (SRC_ROOT, OPENPI_SRC_ROOT, CLIENT_SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


ACTION_DIM = 10


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _image_to_uint8(image: Any) -> np.ndarray:
    image = _to_numpy(image)
    if image.ndim == 4:
        image = image[-1]
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
        if np.nanmax(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return image


def _normalize_rot6d(d6: np.ndarray) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float64)
    a1 = d6[..., :3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def _pose10d_to_mat(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    mats = np.zeros(actions.shape[:-1] + (4, 4), dtype=np.float64)
    mats[..., :3, :3] = _normalize_rot6d(actions[..., 3:9])
    mats[..., :3, 3] = actions[..., :3]
    mats[..., 3, 3] = 1.0
    return mats


def _pose10d_to_rotvec(actions: np.ndarray) -> np.ndarray:
    mats = _pose10d_to_mat(actions)
    return Rotation.from_matrix(mats[..., :3, :3]).as_rotvec()


def _se3_errors(pred_actions: np.ndarray, gt_actions: np.ndarray) -> dict[str, np.ndarray | float]:
    pred_mat = _pose10d_to_mat(pred_actions)
    gt_mat = _pose10d_to_mat(gt_actions)
    delta = np.linalg.inv(gt_mat) @ pred_mat
    trans_error_m = np.linalg.norm(delta[..., :3, 3], axis=-1)
    rot_error_deg = np.rad2deg(Rotation.from_matrix(delta[..., :3, :3]).magnitude())
    gripper_error = pred_actions[..., 9] - gt_actions[..., 9]
    action_abs_error = np.abs(pred_actions - gt_actions)
    return {
        "trans_error_m": trans_error_m,
        "rot_error_deg": rot_error_deg,
        "gripper_error": gripper_error,
        "action_abs_error": action_abs_error,
        "trans_mean_m": float(np.mean(trans_error_m)),
        "trans_rmse_m": float(np.sqrt(np.mean(np.square(trans_error_m)))),
        "trans_max_m": float(np.max(trans_error_m)),
        "rot_mean_deg": float(np.mean(rot_error_deg)),
        "rot_max_deg": float(np.max(rot_error_deg)),
        "gripper_mae": float(np.mean(np.abs(gripper_error))),
        "action_mae": float(np.mean(action_abs_error)),
    }


def _format_metric_text(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"trans mean: {metrics['trans_mean_m'] * 1000.0:.2f} mm",
            f"trans rmse: {metrics['trans_rmse_m'] * 1000.0:.2f} mm",
            f"trans max:  {metrics['trans_max_m'] * 1000.0:.2f} mm",
            f"rot mean:   {metrics['rot_mean_deg']:.2f} deg",
            f"rot max:    {metrics['rot_max_deg']:.2f} deg",
            f"grip MAE:   {metrics['gripper_mae']:.4f}",
            f"action MAE: {metrics['action_mae']:.4f}",
        ]
    )


def _plot_sample(
    *,
    output_path: Path,
    sample_index: int,
    wrist_image: np.ndarray,
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    steps = np.arange(len(gt_actions))
    gt_rotvec = _pose10d_to_rotvec(gt_actions)
    pred_rotvec = _pose10d_to_rotvec(pred_actions)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.05])
    fig.suptitle(f"HandCap train-set action prediction | sample {sample_index}", fontsize=15)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(wrist_image)
    ax_img.set_title("observation.images.wrist")
    ax_img.axis("off")

    ax_pos = fig.add_subplot(gs[0, 1:])
    for dim, name in enumerate(("x", "y", "z")):
        ax_pos.plot(steps, gt_actions[:, dim], linestyle="-", label=f"GT {name}")
        ax_pos.plot(steps, pred_actions[:, dim], linestyle="--", label=f"Pred {name}")
    ax_pos.set_title("relative target translation")
    ax_pos.set_xlabel("action horizon step")
    ax_pos.set_ylabel("meter")
    ax_pos.grid(True, alpha=0.25)
    ax_pos.legend(ncol=3, fontsize=8)

    ax_rot = fig.add_subplot(gs[1, :2])
    for dim, name in enumerate(("rx", "ry", "rz")):
        ax_rot.plot(steps, gt_rotvec[:, dim], linestyle="-", label=f"GT {name}")
        ax_rot.plot(steps, pred_rotvec[:, dim], linestyle="--", label=f"Pred {name}")
    ax_rot.set_title("relative target rotation")
    ax_rot.set_xlabel("action horizon step")
    ax_rot.set_ylabel("rotvec rad")
    ax_rot.grid(True, alpha=0.25)
    ax_rot.legend(ncol=3, fontsize=8)

    ax_err = fig.add_subplot(gs[1, 2])
    ax_err.plot(steps, metrics["trans_error_m"] * 1000.0, label="translation mm")
    ax_err.plot(steps, metrics["rot_error_deg"], label="rotation deg")
    ax_err.set_title("SE(3) error")
    ax_err.set_xlabel("action horizon step")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(fontsize=8)

    ax_3d = fig.add_subplot(gs[2, 0], projection="3d")
    ax_3d.plot(gt_actions[:, 0], gt_actions[:, 1], gt_actions[:, 2], label="GT", linewidth=2.0)
    ax_3d.plot(pred_actions[:, 0], pred_actions[:, 1], pred_actions[:, 2], label="Pred", linewidth=2.0)
    ax_3d.scatter(gt_actions[0, 0], gt_actions[0, 1], gt_actions[0, 2], s=35)
    ax_3d.scatter(pred_actions[0, 0], pred_actions[0, 1], pred_actions[0, 2], s=35)
    ax_3d.set_title("relative target points")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.legend(fontsize=8)

    ax_grip = fig.add_subplot(gs[2, 1])
    ax_grip.plot(steps, gt_actions[:, 9], label="GT gripper")
    ax_grip.plot(steps, pred_actions[:, 9], linestyle="--", label="Pred gripper")
    ax_grip.set_title("gripper")
    ax_grip.set_xlabel("action horizon step")
    ax_grip.grid(True, alpha=0.25)
    ax_grip.legend(fontsize=8)

    ax_text = fig.add_subplot(gs[2, 2])
    ax_text.axis("off")
    ax_text.text(0.02, 0.98, _format_metric_text(metrics), va="top", family="monospace", fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _choose_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    if args.indices:
        indices = [int(x) for x in args.indices.split(",") if x.strip()]
    else:
        start = int(args.start_index)
        stride = int(args.stride)
        indices = [start + i * stride for i in range(args.num_samples)]
    valid = [idx for idx in indices if 0 <= idx < dataset_len]
    if not valid:
        raise ValueError(f"No valid sample indices selected. dataset_len={dataset_len}, requested={indices}")
    return valid


def _build_checkpoint_dir(args: argparse.Namespace, train_config: _config.TrainConfig) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir).expanduser()
    if not args.exp_name or args.checkpoint_step is None:
        raise ValueError("Provide --checkpoint-dir, or provide both --exp-name and --checkpoint-step.")
    return Path(train_config.checkpoint_base_dir) / train_config.name / args.exp_name / str(args.checkpoint_step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="OpenPI training config name, e.g. pi05_513_screw_350.")
    parser.add_argument("--checkpoint-dir", default="", help="Checkpoint/export directory containing params and assets.")
    parser.add_argument("--exp-name", default="", help="Training experiment name, used with --checkpoint-step.")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="Checkpoint step, e.g. 20000.")
    parser.add_argument("--output-dir", default="outputs/handcap_trainset_action_eval", help="Directory for plots/CSV.")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--indices", default="", help="Comma-separated dataset indices. Overrides start/stride.")
    parser.add_argument("--plot-horizon", type=int, default=20, help="Number of action steps to plot/score.")
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--pytorch-device", default=None, help="Optional torch device for PyTorch checkpoints.")
    parser.add_argument("--save-npz", action="store_true", help="Save GT/pred arrays for each plotted sample.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    checkpoint_dir = _build_checkpoint_dir(args, train_config)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )
    repack = transforms.compose(data_config.repack_transforms.inputs)
    indices = _choose_indices(len(dataset), args)
    rows: list[dict[str, Any]] = []

    for sample_index in indices:
        sample = dataset[sample_index]
        repacked = repack(sample)
        gt_actions = _to_numpy(repacked.pop("actions")).astype(np.float32)
        policy_obs = dict(repacked)
        pred_actions = _to_numpy(policy.infer(policy_obs)["actions"]).astype(np.float32)

        horizon = min(args.plot_horizon, gt_actions.shape[0], pred_actions.shape[0])
        gt = gt_actions[:horizon, :ACTION_DIM]
        pred = pred_actions[:horizon, :ACTION_DIM]
        metrics = _se3_errors(pred, gt)
        wrist = _image_to_uint8(policy_obs["observation/wrist_image"])
        plot_path = output_dir / f"sample_{sample_index:06d}.png"
        _plot_sample(
            output_path=plot_path,
            sample_index=sample_index,
            wrist_image=wrist,
            gt_actions=gt,
            pred_actions=pred,
            metrics=metrics,
        )

        if args.save_npz:
            np.savez_compressed(
                output_dir / f"sample_{sample_index:06d}.npz",
                gt_actions=gt,
                pred_actions=pred,
                trans_error_m=metrics["trans_error_m"],
                rot_error_deg=metrics["rot_error_deg"],
                gripper_error=metrics["gripper_error"],
            )

        row = {
            "sample_index": sample_index,
            "horizon": horizon,
            "plot": str(plot_path),
            "trans_mean_mm": metrics["trans_mean_m"] * 1000.0,
            "trans_rmse_mm": metrics["trans_rmse_m"] * 1000.0,
            "trans_max_mm": metrics["trans_max_m"] * 1000.0,
            "rot_mean_deg": metrics["rot_mean_deg"],
            "rot_max_deg": metrics["rot_max_deg"],
            "gripper_mae": metrics["gripper_mae"],
            "action_mae": metrics["action_mae"],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True))

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {csv_path}")


if __name__ == "__main__":
    main()
