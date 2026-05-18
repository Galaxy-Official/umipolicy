#!/usr/bin/env python3
"""Replay one HandCap training episode through a checkpoint and save an action-comparison video."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import cv2
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


def _scalar(value: Any) -> int:
    arr = _to_numpy(value)
    return int(arr.item() if arr.shape == () else arr.reshape(-1)[0])


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
    return image.astype(np.uint8, copy=False)


def _unwrap_dataset(dataset: Any) -> Any:
    while hasattr(dataset, "_dataset"):
        dataset = dataset._dataset
    return dataset


def _episode_bounds(base_dataset: Any, episode_index: int) -> tuple[int, int]:
    ep = base_dataset.meta.episodes[episode_index]
    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


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


def _pose6_to_mat(pose6: np.ndarray) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64)
    mats = np.zeros(pose6.shape[:-1] + (4, 4), dtype=np.float64)
    mats[..., :3, 3] = pose6[..., :3]
    mats[..., :3, :3] = Rotation.from_rotvec(pose6[..., 3:6]).as_matrix()
    mats[..., 3, 3] = 1.0
    return mats


def _mat_to_pose6(mats: np.ndarray) -> np.ndarray:
    mats = np.asarray(mats, dtype=np.float64)
    pos = mats[..., :3, 3]
    rotvec = Rotation.from_matrix(mats[..., :3, :3]).as_rotvec()
    return np.concatenate([pos, rotvec], axis=-1)


def _relative_10d_to_abs_pose7(rel_actions: np.ndarray, current_abs_pose6: np.ndarray) -> np.ndarray:
    rel_mats = _pose10d_to_mat(rel_actions)
    current_mat = _pose6_to_mat(current_abs_pose6)
    abs_mats = current_mat @ rel_mats
    abs_pose6 = _mat_to_pose6(abs_mats)
    gripper = rel_actions[..., 9:10]
    return np.concatenate([abs_pose6, gripper], axis=-1)


def _raw_abs_action_to_pose7(raw_actions: np.ndarray) -> np.ndarray:
    raw_actions = np.asarray(raw_actions, dtype=np.float64)
    return np.concatenate([raw_actions[..., :6], raw_actions[..., 6:7]], axis=-1)


def _se3_error(pred_pose7: np.ndarray, gt_pose7: np.ndarray) -> dict[str, np.ndarray | float]:
    pred_mat = _pose6_to_mat(pred_pose7[..., :6])
    gt_mat = _pose6_to_mat(gt_pose7[..., :6])
    delta = np.linalg.inv(gt_mat) @ pred_mat
    trans_error_m = np.linalg.norm(delta[..., :3, 3], axis=-1)
    rot_error_deg = np.rad2deg(Rotation.from_matrix(delta[..., :3, :3]).magnitude())
    gripper_error = pred_pose7[..., 6] - gt_pose7[..., 6]
    return {
        "trans_error_m": trans_error_m,
        "rot_error_deg": rot_error_deg,
        "gripper_error": gripper_error,
        "trans_mean_m": float(np.mean(trans_error_m)),
        "trans_max_m": float(np.max(trans_error_m)),
        "rot_mean_deg": float(np.mean(rot_error_deg)),
        "rot_max_deg": float(np.max(rot_error_deg)),
        "gripper_mae": float(np.mean(np.abs(gripper_error))),
    }


def _build_checkpoint_dir(args: argparse.Namespace, train_config: _config.TrainConfig) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir).expanduser()
    if not args.exp_name or args.checkpoint_step is None:
        raise ValueError("Provide --checkpoint-dir, or both --exp-name and --checkpoint-step.")
    return Path(train_config.checkpoint_base_dir) / train_config.name / args.exp_name / str(args.checkpoint_step)


def _select_indices(start: int, end: int, args: argparse.Namespace) -> list[int]:
    indices = list(range(start + args.start_offset, end, args.frame_stride))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]
    if not indices:
        raise ValueError(f"No frames selected from episode range [{start}, {end}).")
    return indices


def _collect_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    base_dataset = _unwrap_dataset(dataset)
    reader = base_dataset._ensure_reader()
    checkpoint_dir = _build_checkpoint_dir(args, train_config)

    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )
    repack = transforms.compose(data_config.repack_transforms.inputs)
    ep_start, ep_end = _episode_bounds(base_dataset, args.episode_index)
    indices = _select_indices(ep_start, ep_end, args)
    records: list[dict[str, Any]] = []

    for count, dataset_index in enumerate(indices):
        raw = reader.get_item(dataset_index)
        processed = dataset[dataset_index]
        repacked = repack(processed)
        repacked.pop("actions", None)

        policy_result = policy.infer(dict(repacked))
        pred_rel = _to_numpy(policy_result["actions"]).astype(np.float64)
        raw_state = _to_numpy(raw["observation.state"]).astype(np.float64)
        raw_actions = _to_numpy(raw["action"]).astype(np.float64)

        horizon = min(args.plot_horizon, pred_rel.shape[0], raw_actions.shape[0])
        current_pose6 = raw_state[:6]
        current_pose7 = np.concatenate([raw_state[:6], raw_state[6:7]])
        pred_abs = _relative_10d_to_abs_pose7(pred_rel[:horizon, :ACTION_DIM], current_pose6)
        gt_abs = _raw_abs_action_to_pose7(raw_actions[:horizon])
        chunk_metrics = _se3_error(pred_abs, gt_abs)
        first_metrics = _se3_error(pred_abs[:1], gt_abs[:1])

        record = {
            "dataset_index": int(dataset_index),
            "episode_index": _scalar(raw["episode_index"]),
            "frame_index": _scalar(raw["frame_index"]),
            "timestamp": float(_to_numpy(raw["timestamp"]).reshape(-1)[0]),
            "wrist_image": _image_to_uint8(repacked["observation/wrist_image"]),
            "current_pose7": current_pose7,
            "pred_abs": pred_abs,
            "gt_abs": gt_abs,
            "chunk_metrics": chunk_metrics,
            "first_metrics": first_metrics,
        }
        records.append(record)

        print(
            json.dumps(
                {
                    "progress": f"{count + 1}/{len(indices)}",
                    "dataset_index": record["dataset_index"],
                    "frame_index": record["frame_index"],
                    "first_trans_mm": first_metrics["trans_mean_m"] * 1000.0,
                    "first_rot_deg": first_metrics["rot_mean_deg"],
                    "chunk_trans_mean_mm": chunk_metrics["trans_mean_m"] * 1000.0,
                    "chunk_rot_mean_deg": chunk_metrics["rot_mean_deg"],
                },
                ensure_ascii=True,
            )
        )

    context = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "episode_index": args.episode_index,
        "episode_range": [ep_start, ep_end],
        "fps": float(getattr(base_dataset, "fps", args.video_fps)),
        "num_records": len(records),
    }
    return records, context


def _axis_limits(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for rec in records:
        points.append(rec["current_pose7"][None, :3])
        points.append(rec["pred_abs"][:, :3])
        points.append(rec["gt_abs"][:, :3])
    pts = np.concatenate(points, axis=0)
    center = (np.nanmin(pts, axis=0) + np.nanmax(pts, axis=0)) / 2.0
    span = np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0)
    radius = max(float(np.nanmax(span)) / 2.0, 0.02)
    return center - radius, center + radius


def _metric_text(rec: dict[str, Any]) -> str:
    first = rec["first_metrics"]
    chunk = rec["chunk_metrics"]
    return "\n".join(
        [
            f"episode: {rec['episode_index']}",
            f"frame:   {rec['frame_index']}",
            f"time:    {rec['timestamp']:.3f}s",
            "",
            "first action",
            f"  trans: {first['trans_mean_m'] * 1000.0:.2f} mm",
            f"  rot:   {first['rot_mean_deg']:.2f} deg",
            f"  grip:  {first['gripper_mae']:.4f}",
            "",
            "chunk mean",
            f"  trans: {chunk['trans_mean_m'] * 1000.0:.2f} mm",
            f"  rot:   {chunk['rot_mean_deg']:.2f} deg",
            f"  grip:  {chunk['gripper_mae']:.4f}",
        ]
    )


def _render_frame(records: list[dict[str, Any]], frame_idx: int, xyz_min: np.ndarray, xyz_max: np.ndarray) -> np.ndarray:
    rec = records[frame_idx]
    history = records[: frame_idx + 1]
    actual_hist = np.stack([r["current_pose7"][:3] for r in history], axis=0)
    gt_first_hist = np.stack([r["gt_abs"][0, :3] for r in history], axis=0)
    pred_first_hist = np.stack([r["pred_abs"][0, :3] for r in history], axis=0)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.35, 1.05], height_ratios=[1.0, 1.0])
    fig.suptitle("Offline real-inference replay on one training episode", fontsize=15)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(rec["wrist_image"])
    ax_img.set_title("observation.images.wrist")
    ax_img.axis("off")

    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")
    ax_3d.plot(actual_hist[:, 0], actual_hist[:, 1], actual_hist[:, 2], color="0.15", label="GT observed state")
    ax_3d.plot(gt_first_hist[:, 0], gt_first_hist[:, 1], gt_first_hist[:, 2], color="#1f77b4", label="GT next target")
    ax_3d.plot(pred_first_hist[:, 0], pred_first_hist[:, 1], pred_first_hist[:, 2], color="#d62728", label="Pred next target")
    ax_3d.plot(rec["gt_abs"][:, 0], rec["gt_abs"][:, 1], rec["gt_abs"][:, 2], "--", color="#1f77b4", alpha=0.7, label="GT chunk")
    ax_3d.plot(rec["pred_abs"][:, 0], rec["pred_abs"][:, 1], rec["pred_abs"][:, 2], "--", color="#d62728", alpha=0.7, label="Pred chunk")
    ax_3d.scatter(*rec["current_pose7"][:3], color="0.05", s=45)
    ax_3d.scatter(*rec["gt_abs"][0, :3], color="#1f77b4", s=45)
    ax_3d.scatter(*rec["pred_abs"][0, :3], color="#d62728", s=45)
    ax_3d.set_xlim(xyz_min[0], xyz_max[0])
    ax_3d.set_ylim(xyz_min[1], xyz_max[1])
    ax_3d.set_zlim(xyz_min[2], xyz_max[2])
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.set_title("absolute target trajectory")
    ax_3d.legend(fontsize=8, loc="upper left")

    steps = np.arange(len(rec["gt_abs"]))
    ax_xyz = fig.add_subplot(gs[0, 2])
    labels = ("x", "y", "z")
    for dim, label in enumerate(labels):
        ax_xyz.plot(steps, rec["gt_abs"][:, dim], color=f"C{dim}", linestyle="-", label=f"GT {label}")
        ax_xyz.plot(steps, rec["pred_abs"][:, dim], color=f"C{dim}", linestyle="--", label=f"Pred {label}")
    ax_xyz.set_title("current-frame action chunk xyz")
    ax_xyz.set_xlabel("horizon step")
    ax_xyz.set_ylabel("meter")
    ax_xyz.grid(True, alpha=0.25)
    ax_xyz.legend(fontsize=7, ncol=2)

    ax_err = fig.add_subplot(gs[1, 0])
    chunk = rec["chunk_metrics"]
    ax_err.plot(steps, chunk["trans_error_m"] * 1000.0, label="translation mm")
    ax_err.plot(steps, chunk["rot_error_deg"], label="rotation deg")
    ax_err.set_title("chunk SE(3) error")
    ax_err.set_xlabel("horizon step")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(fontsize=8)

    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis("off")
    ax_text.text(0.0, 1.0, _metric_text(rec), va="top", family="monospace", fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return frame


def _write_video(records: list[dict[str, Any]], output_path: Path, fps: float) -> None:
    xyz_min, xyz_max = _axis_limits(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for idx in range(len(records)):
            frame = _render_frame(records, idx, xyz_min, xyz_max)
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {output_path}")
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()


def _write_summary(records: list[dict[str, Any]], context: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in records:
        first = rec["first_metrics"]
        chunk = rec["chunk_metrics"]
        rows.append(
            {
                "dataset_index": rec["dataset_index"],
                "episode_index": rec["episode_index"],
                "frame_index": rec["frame_index"],
                "timestamp": rec["timestamp"],
                "first_trans_mm": first["trans_mean_m"] * 1000.0,
                "first_rot_deg": first["rot_mean_deg"],
                "first_gripper_mae": first["gripper_mae"],
                "chunk_trans_mean_mm": chunk["trans_mean_m"] * 1000.0,
                "chunk_trans_max_mm": chunk["trans_max_m"] * 1000.0,
                "chunk_rot_mean_deg": chunk["rot_mean_deg"],
                "chunk_rot_max_deg": chunk["rot_max_deg"],
                "chunk_gripper_mae": chunk["gripper_mae"],
            }
        )
    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(context, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="OpenPI training config name, e.g. pi05_513_screw_350.")
    parser.add_argument("--checkpoint-dir", default="", help="Checkpoint/export directory containing params and assets.")
    parser.add_argument("--exp-name", default="", help="Training experiment name, used with --checkpoint-step.")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="Checkpoint step, e.g. 20000.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/handcap_episode_replay", help="Directory for video/CSV.")
    parser.add_argument("--output-video", default="", help="Optional explicit mp4 path.")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means use the whole selected episode.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--plot-horizon", type=int, default=20)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--pytorch-device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    records, context = _collect_records(args)
    video_path = Path(args.output_video).expanduser() if args.output_video else output_dir / "episode_replay.mp4"
    context["output_video"] = str(video_path)
    _write_video(records, video_path, args.video_fps)
    _write_summary(records, context, output_dir)
    print(f"Saved video: {video_path}")
    print(f"Saved summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
