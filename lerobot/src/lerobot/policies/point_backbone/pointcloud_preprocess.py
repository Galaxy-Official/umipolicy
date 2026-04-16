#!/usr/bin/env python3
"""
Generate per-frame point clouds from a LeRobot-style `handcap_phone` dataset
with the newer layout where phone depth is stored inside parquet as:

    observation.images.phone_depth: struct<bytes: binary, path: string>

Supported dataset conventions
-----------------------------
Newer layout (from uploaded info.json):
- parquet files:
    data/chunk-000/file-000.parquet
- phone videos:
    videos/observation.images.phone/chunk-000/file-000.mp4
- depth source:
    parquet column `observation.images.phone_depth`

Legacy layout (kept as a fallback):
- parquet files:
    data/chunk-000/episode_000000.parquet
- phone videos:
    videos/chunk-000/observation.images.phone/episode_000000.mp4
- depth source:
    images/observation.images.phone_depth/episode_000000/frame_000000.png

Output layout
-------------
handcap_phone/
└── pointclouds/
    └── observation.points.phone/
        ├── episode_000000/
        │   ├── frame_000000.npz   # keys: xyz, rgb, mask
        │   └── ...
        └── metadata/
            └── pointcloud_info.json

Each .npz contains:
- xyz:  (N, 3) float32, phone-camera point cloud in meters
- rgb:  (N, 3) uint8
- mask: empty bool array by default

Notes
-----
1. This script does NOT modify parquet / info.json.
2. Point clouds are generated in the phone camera coordinate system.
3. If the parquet depth struct contains `bytes`, they are decoded first.
4. If `bytes` is empty but `path` is present, the path is used as a fallback.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError as exc:
    raise SystemExit("This script requires opencv-python. Please install cv2 first.") from exc

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("This script requires Pillow. Please install pillow first.") from exc


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class EpisodeStats:
    episode: str
    total_rows: int
    saved_frames: int
    skipped_missing_depth: int
    skipped_invalid_depth: int
    skipped_missing_video: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate phone-camera point clouds from handcap_phone RGB video + depth stored in parquet struct bytes/path."
        )
    )
    parser.add_argument("--handcap_phone", type=str, required=True, help="Path to the handcap_phone dataset root.")
    parser.add_argument("--fx", type=float, required=True, help="Camera intrinsic fx.")
    parser.add_argument("--fy", type=float, required=True, help="Camera intrinsic fy.")
    parser.add_argument("--cx", type=float, required=True, help="Camera intrinsic cx.")
    parser.add_argument("--cy", type=float, required=True, help="Camera intrinsic cy.")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help=(
            "Divisor used to convert raw depth to meters. Default 1000.0 means uint16 depth in millimeters. "
            "Set to 1.0 if your depth data already stores meters."
        ),
    )
    parser.add_argument("--depth-min", type=float, default=1e-4, help="Minimum valid depth in meters.")
    parser.add_argument("--depth-max", type=float, default=10.0, help="Maximum valid depth in meters.")
    parser.add_argument("--pixel-stride", type=int, default=1, help="Subsample image grid by this stride.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="If > 0, uniformly subsample each frame point cloud to at most this many points.",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="",
        help="Optional comma-separated episode ids like 000000,000001. Empty means all episodes.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz pointcloud files.")
    return parser.parse_args()


class VideoFrameReader:
    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise RuntimeError(f"Video not found: {self.video_path}")

        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames",
            "-of", "json",
            str(self.video_path),
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        meta = json.loads(result.stdout)
        stream = meta["streams"][0]
        self.width = int(stream["width"])
        self.height = int(stream["height"])

    def read_frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            raise ValueError(f"frame_index must be >= 0, got {frame_index}")

        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", str(self.video_path),
            "-vf", f"select=eq(n\\,{frame_index})",
            "-vframes", "1",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw = result.stdout

        expected = self.width * self.height * 3
        if len(raw) != expected:
            err = result.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Failed to decode frame {frame_index} from {self.video_path}\n{err}"
            )

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        return frame

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _decode_depth_image_bytes(raw: bytes | bytearray | memoryview, depth_scale: float) -> np.ndarray:
    raw_bytes = bytes(raw)
    if len(raw_bytes) == 0:
        raise ValueError("Depth bytes are empty.")

    # Try Pillow first because parquet often stores PNG bytes.
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        depth_img = np.array(img)
    except Exception:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        depth_img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise ValueError("Failed to decode depth bytes as image.")

    if np.issubdtype(depth_img.dtype, np.integer):
        return depth_img.astype(np.float32) / float(depth_scale)

    depth_m = depth_img.astype(np.float32)
    if not math.isclose(depth_scale, 1.0):
        depth_m = depth_m / float(depth_scale)
    return depth_m


def _read_depth_from_path(path: Path, depth_scale: float) -> np.ndarray:
    depth_img = np.array(Image.open(path))
    if np.issubdtype(depth_img.dtype, np.integer):
        return depth_img.astype(np.float32) / float(depth_scale)

    depth_m = depth_img.astype(np.float32)
    if not math.isclose(depth_scale, 1.0):
        depth_m = depth_m / float(depth_scale)
    return depth_m


def _extract_depth_struct(cell: Any) -> tuple[bytes | None, str | None]:
    """Extract (bytes, path) from a parquet struct cell as robustly as possible."""
    if cell is None:
        return None, None

    if isinstance(cell, dict):
        return cell.get("bytes"), cell.get("path")

    if hasattr(cell, "as_py"):
        py_obj = cell.as_py()
        if isinstance(py_obj, dict):
            return py_obj.get("bytes"), py_obj.get("path")

    # pandas may materialize struct as tuple/list in field order
    if isinstance(cell, (tuple, list)) and len(cell) >= 2:
        return cell[0], cell[1]

    # pyarrow scalar sometimes exposes fields by name
    try:
        raw_bytes = cell["bytes"]
        raw_path = cell["path"]
        if hasattr(raw_bytes, "as_py"):
            raw_bytes = raw_bytes.as_py()
        if hasattr(raw_path, "as_py"):
            raw_path = raw_path.as_py()
        return raw_bytes, raw_path
    except Exception:
        pass

    raise TypeError(f"Unsupported depth struct cell type: {type(cell)}")


def load_depth_from_cell(cell: Any, dataset_root: Path, depth_scale: float) -> np.ndarray:
    raw_bytes, raw_path = _extract_depth_struct(cell)

    if raw_bytes is not None:
        if isinstance(raw_bytes, memoryview):
            raw_bytes = raw_bytes.tobytes()
        if isinstance(raw_bytes, bytearray):
            raw_bytes = bytes(raw_bytes)
        if isinstance(raw_bytes, bytes) and len(raw_bytes) > 0:
            return _decode_depth_image_bytes(raw_bytes, depth_scale=depth_scale)

    if raw_path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        if candidate.is_file():
            return _read_depth_from_path(candidate, depth_scale=depth_scale)
        raise FileNotFoundError(f"Depth path from parquet does not exist: {candidate}")

    raise FileNotFoundError("Depth struct contains neither usable bytes nor a valid path.")


def backproject_depth_to_pointcloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intr: CameraIntrinsics,
    depth_min: float,
    depth_max: float,
    pixel_stride: int = 1,
    max_points: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if depth_m.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {depth_m.shape}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
    if depth_m.shape != rgb.shape[:2]:
        raise ValueError(f"depth and rgb resolution mismatch: depth={depth_m.shape}, rgb={rgb.shape[:2]}")
    if pixel_stride <= 0:
        raise ValueError(f"pixel_stride must be >= 1, got {pixel_stride}")

    depth = depth_m[::pixel_stride, ::pixel_stride]
    color = rgb[::pixel_stride, ::pixel_stride, :]
    h, w = depth.shape

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xs = xs.astype(np.float32) * float(pixel_stride)
    ys = ys.astype(np.float32) * float(pixel_stride)

    valid = np.isfinite(depth)
    valid &= depth > depth_min
    valid &= depth < depth_max

    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    z = depth[valid].astype(np.float32)
    x = (xs[valid] - float(intr.cx)) * z / float(intr.fx)
    y = (ys[valid] - float(intr.cy)) * z / float(intr.fy)

    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    rgb_valid = color[valid].reshape(-1, 3).astype(np.uint8)

    if max_points > 0 and xyz.shape[0] > max_points:
        idx = np.linspace(0, xyz.shape[0] - 1, num=max_points, dtype=np.int64)
        xyz = xyz[idx]
        rgb_valid = rgb_valid[idx]

    return xyz, rgb_valid


def _find_phone_video_for_parquet(root: Path, parquet_path: Path) -> Path | None:
    chunk_name = parquet_path.parent.name
    stem = parquet_path.stem

    # New layout from info.json
    new_path = root / "videos" / "observation.images.phone" / chunk_name / f"{stem}.mp4"
    if new_path.is_file():
        return new_path

    # Legacy layout
    legacy_path = root / "videos" / chunk_name / "observation.images.phone" / f"{stem}.mp4"
    if legacy_path.is_file():
        return legacy_path

    return None


def discover_parquet_sources(root: Path) -> Iterator[tuple[Path, Path | None]]:
    data_root = root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing data directory: {data_root}")

    parquet_paths = sorted(data_root.glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under: {data_root}")

    for parquet_path in parquet_paths:
        yield parquet_path, _find_phone_video_for_parquet(root, parquet_path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_parquet_file(
    parquet_path: Path,
    video_path: Path | None,
    out_root: Path,
    intr: CameraIntrinsics,
    args: argparse.Namespace,
    dataset_root: Path,
    requested: set[str] | None,
) -> list[EpisodeStats]:
    required_cols = ["frame_index", "episode_index", "observation.images.phone_depth"]
    df = pd.read_parquet(parquet_path, columns=required_cols)
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in {parquet_path}: {missing_cols}")

    stats_map: dict[str, EpisodeStats] = {}

    if video_path is None or not video_path.is_file():
        for episode_index, group in df.groupby("episode_index", sort=False):
            episode_name = f"episode_{int(episode_index):06d}"
            if requested is not None and episode_name.split("_")[-1] not in requested:
                continue
            stats_map[episode_name] = EpisodeStats(
                episode=episode_name,
                total_rows=len(group),
                saved_frames=0,
                skipped_missing_depth=0,
                skipped_invalid_depth=0,
                skipped_missing_video=len(group),
            )
        return list(stats_map.values())

    with VideoFrameReader(video_path) as reader:
        for _, row in df.iterrows():
            episode_index = int(row["episode_index"])
            frame_index = int(row["frame_index"])
            episode_name = f"episode_{episode_index:06d}"
            episode_id = episode_name.split("_")[-1]

            if requested is not None and episode_id not in requested:
                continue

            if episode_name not in stats_map:
                stats_map[episode_name] = EpisodeStats(
                    episode=episode_name,
                    total_rows=0,
                    saved_frames=0,
                    skipped_missing_depth=0,
                    skipped_invalid_depth=0,
                    skipped_missing_video=0,
                )
            stats_map[episode_name].total_rows += 1

            out_dir = out_root / episode_name
            ensure_dir(out_dir)
            save_path = out_dir / f"frame_{frame_index:06d}.npz"
            if save_path.exists() and not args.overwrite:
                stats_map[episode_name].saved_frames += 1
                continue

            try:
                depth_m = load_depth_from_cell(
                    row["observation.images.phone_depth"],
                    dataset_root=dataset_root,
                    depth_scale=args.depth_scale,
                )
            except FileNotFoundError:
                stats_map[episode_name].skipped_missing_depth += 1
                continue
            except Exception:
                stats_map[episode_name].skipped_invalid_depth += 1
                continue

            rgb = reader.read_frame(frame_index)
            if depth_m.shape != rgb.shape[:2]:
                depth_m = cv2.resize(depth_m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            xyz, rgb_valid = backproject_depth_to_pointcloud(
                depth_m=depth_m,
                rgb=rgb,
                intr=intr,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                pixel_stride=args.pixel_stride,
                max_points=args.max_points,
            )
            if xyz.shape[0] == 0:
                stats_map[episode_name].skipped_invalid_depth += 1
                continue

            empty_mask = np.empty((0,), dtype=bool)
            np.savez_compressed(save_path, xyz=xyz, rgb=rgb_valid, mask=empty_mask)
            stats_map[episode_name].saved_frames += 1

    return list(stats_map.values())


def main() -> None:
    args = parse_args()
    root = Path(args.handcap_phone).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"handcap_phone root not found: {root}")

    intr = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
    requested = None
    if args.episodes.strip():
        requested = {item.strip() for item in args.episodes.split(",") if item.strip()}

    output_root = root / "pointclouds" / "observation.points.phone"
    metadata_dir = output_root / "metadata"
    ensure_dir(metadata_dir)

    all_stats: list[EpisodeStats] = []
    for parquet_path, video_path in discover_parquet_sources(root):
        print(f"[process] parquet={parquet_path.name}")
        if video_path is None:
            print("  warning: phone video not found for this parquet; its rows will be counted as skipped_missing_video")
        else:
            print(f"  video={video_path}")

        file_stats = process_parquet_file(
            parquet_path=parquet_path,
            video_path=video_path,
            out_root=output_root,
            intr=intr,
            args=args,
            dataset_root=root,
            requested=requested,
        )
        all_stats.extend(file_stats)

        for item in file_stats:
            print(
                f"  {item.episode}: saved={item.saved_frames}, missing_depth={item.skipped_missing_depth}, "
                f"invalid_depth={item.skipped_invalid_depth}, missing_video={item.skipped_missing_video}"
            )

    summary = {
        "handcap_phone": str(root),
        "output_root": str(output_root),
        "depth_source": "parquet struct observation.images.phone_depth {bytes, path}",
        "storage_format": {
            "per_frame_file": "npz",
            "keys": {
                "xyz": "(N, 3) float32, meters, phone camera coordinates",
                "rgb": "(N, 3) uint8, aligned per-point RGB",
                "mask": "empty bool array by default",
            },
        },
        "camera_intrinsics": asdict(intr),
        "depth_settings": {
            "depth_scale": args.depth_scale,
            "depth_min": args.depth_min,
            "depth_max": args.depth_max,
            "pixel_stride": args.pixel_stride,
            "max_points": args.max_points,
        },
        "episodes": [asdict(item) for item in all_stats],
    }
    with open(metadata_dir / "pointcloud_info.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[done] point cloud preprocessing finished.")
    print(f"[done] outputs saved under: {output_root}")


if __name__ == "__main__":
    main()
