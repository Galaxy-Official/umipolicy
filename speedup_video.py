#!/usr/bin/env python3
"""Speed up videos with ffmpeg and save MP4 outputs.

Examples:
    python speedup_video.py input.mov
    python speedup_video.py input.avi --speed 3
    python speedup_video.py input.mp4 --speed 5 --output input_fast.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speed up video files and save MP4 outputs named like input_5x.mp4.")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input video file(s). Any format readable by ffmpeg is supported.")
    parser.add_argument(
        "-s",
        "--speed",
        type=float,
        default=5.0,
        help="Speed multiplier. For example: 5 means 5x, 3 means 3x. Default: 5.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output mp4 path. Only allowed when processing one input.")
    parser.add_argument(
        "-y",
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.")
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality value. Lower is higher quality/larger file. Default: 18.")
    parser.add_argument(
        "--preset",
        default="medium",
        help="ffmpeg x264 preset. Use faster/veryfast for speed, slow for smaller files.")
    return parser.parse_args()


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"Error: {name} not found in PATH. Please install ffmpeg first.")
    return path


def format_speed_for_name(speed: float) -> str:
    if speed.is_integer():
        return str(int(speed))
    return f"{speed:g}".replace(".", "p")


def default_output_path(input_path: Path, speed: float) -> Path:
    speed_name = format_speed_for_name(speed)
    return input_path.with_name(f"{input_path.stem}_{speed_name}x.mp4")


def has_audio_stream(ffprobe: str, input_path: Path) -> bool:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        str(input_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {input_path}:\n{proc.stderr.strip()}")
    data = json.loads(proc.stdout or "{}")
    return bool(data.get("streams"))


def atempo_filter(speed: float) -> str:
    """Build an ffmpeg atempo chain.

    Keep factors in [0.5, 2.0] for compatibility with older ffmpeg versions.
    """
    factors: list[float] = []
    remaining = speed

    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0

    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5

    factors.append(remaining)
    return ",".join(f"atempo={factor:g}" for factor in factors)


def build_ffmpeg_command(
        ffmpeg: str,
        input_path: Path,
        output_path: Path,
        speed: float,
        with_audio: bool,
        overwrite: bool,
        crf: int,
        preset: str) -> list[str]:
    overwrite_flag = "-y" if overwrite else "-n"

    base = [
        ffmpeg,
        overwrite_flag,
        "-hide_banner",
        "-i",
        str(input_path),
    ]

    encode = [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "0",
        str(output_path),
    ]

    if with_audio:
        filter_complex = (
            f"[0:v:0]setpts=PTS/{speed:g}[v];"
            f"[0:a:0]{atempo_filter(speed)}[a]")
        return base + [
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ] + encode

    return base + [
        "-map",
        "0:v:0",
        "-filter:v",
        f"setpts=PTS/{speed:g}",
        "-an",
    ] + encode


def speedup_one(
        ffmpeg: str,
        ffprobe: str,
        input_path: Path,
        output_path: Path,
        speed: float,
        overwrite: bool,
        crf: int,
        preset: str) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must be different from input path.")

    audio = has_audio_stream(ffprobe, input_path)
    cmd = build_ffmpeg_command(
        ffmpeg=ffmpeg,
        input_path=input_path,
        output_path=output_path,
        speed=speed,
        with_audio=audio,
        overwrite=overwrite,
        crf=crf,
        preset=preset,
    )

    print(f"[speedup] {input_path} -> {output_path} ({speed:g}x)")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}: {input_path}")


def main() -> int:
    args = parse_args()
    if args.speed <= 0:
        raise SystemExit("Error: --speed must be greater than 0.")
    if args.output is not None and len(args.inputs) != 1:
        raise SystemExit("Error: --output can only be used with one input file.")

    ffmpeg = require_tool("ffmpeg")
    ffprobe = require_tool("ffprobe")

    try:
        for input_path in args.inputs:
            output_path = args.output if args.output is not None else default_output_path(
                input_path, args.speed)
            speedup_one(
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                input_path=input_path,
                output_path=output_path,
                speed=args.speed,
                overwrite=args.overwrite,
                crf=args.crf,
                preset=args.preset,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
