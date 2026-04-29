#!/usr/bin/env python3
"""Monitor training jobs listed in monitor_input.json.

The monitor reloads its input file every polling cycle, so tasks can be added
or removed without restarting this script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob as globlib
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "monitor_input.json"
DEFAULT_OUTPUT = ROOT / "monitor_output" / "state_sum.json"

ERROR_RE = re.compile(
    r"("
    r"Traceback \(most recent call last\)|"
    r"\b(?:RuntimeError|ValueError|KeyError|Exception|ChildFailedError|OutOfMemoryError|"
    r"CUDA out of memory|Segmentation fault|Aborted|Killed|NaN|nan loss)\b|"
    r"(?:^|\s)(?:ERROR|Error|FAILED|Failed|failed)(?:\b|:)"
    r")"
)

PROGRESS_RES = [
    re.compile(r"Progress on:\s*([0-9.]+)\s*k?it\s*/\s*([0-9.]+)\s*k?it", re.I),
    re.compile(r"\bStep\s+0*([0-9]+)\b", re.I),
    re.compile(r"\b(?:global_step|train_step|step)\s*[=:]\s*0*([0-9]+)\b", re.I),
    re.compile(r"\bEpoch\s+0*([0-9]+)(?:\s*/\s*0*([0-9]+))?", re.I),
    re.compile(r"(\d{1,3})%\|.*?\|\s*([0-9]+)\s*/\s*([0-9]+)"),
]

CAPTURE_FULL_RE = re.compile(r"exec\s+>\s*>\(\s*tee\b", re.M)


def local_now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def iso(ts: float | None = None) -> str:
    if ts is None:
        return local_now().isoformat()
    return dt.datetime.fromtimestamp(ts).astimezone().isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "task"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def ensure_input(path: Path) -> None:
    if path.exists():
        return
    template = {
        "_notes": [
            "Add or remove tasks while monitor_training.py is running; it reloads this file every cycle.",
            "Each task can be a relative script path string or an object with name/script/enabled/log_globs/checkpoint_globs.",
        ],
        "poll_interval_seconds": 60,
        "stale_after_seconds": 1800,
        "tasks": [
            {
                "name": "umi_erase_board",
                "script": "umi/run_scripts/train/run_umi_train_429_erase_board.sh",
                "enabled": True,
            },
            {
                "name": "openpi_erase_board_wrist_pi05",
                "script": "openpi/scripts/handcap_training/erase_board_wrist_pi05.sh",
                "enabled": True,
            },
            {
                "name": "cosmos_flexiv",
                "script": "cosmos-policy/scripts/flexiv_training/run_train.sh",
                "enabled": True,
            },
            {
                "name": "lingbot_flexiv",
                "script": "lingbot-va/script/run_va_posttrain_flexiv.sh",
                "enabled": True,
            },
        ],
    }
    write_json_atomic(path, template)


def normalize_task(raw: Any, index: int) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"name": Path(raw).stem, "script": raw, "enabled": True}
    if not isinstance(raw, dict):
        return {"name": f"invalid_task_{index}", "script": "", "enabled": False, "invalid": raw}
    task = dict(raw)
    task.setdefault("name", Path(str(task.get("script", f"task_{index}"))).stem)
    task.setdefault("enabled", True)
    return task


def task_repo_root(script_rel: str, root: Path) -> Path:
    parts = Path(script_rel).parts
    if parts:
        return root / parts[0]
    return root


def glob_existing(root: Path, patterns: list[str], limit: int = 3000) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        base = Path(pattern)
        raw_matches = globlib.glob(pattern, recursive=True) if base.is_absolute() else root.glob(pattern)
        matches = (Path(match) for match in raw_matches)
        for match in matches:
            try:
                resolved = match.resolve()
            except OSError:
                continue
            if resolved in seen or not match.exists():
                continue
            seen.add(resolved)
            out.append(match)
            if len(out) >= limit:
                return out
    return out


def newest_path(paths: list[Path]) -> Path | None:
    newest: tuple[float, Path] | None = None
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest[0]:
            newest = (mtime, path)
    return newest[1] if newest else None


def read_tail(path: Path, max_bytes: int = 240_000) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def default_log_globs(script_rel: str, task_name: str) -> list[str]:
    script = Path(script_rel)
    stem = script.stem
    parent_stem = f"{script.parent.name}_{stem}" if script.parent.name else stem
    repo = script.parts[0] if script.parts else ""
    globs = [
        f"logs/{stem}_*.log",
        f"logs/{parent_stem}_*.log",
        f"logs/*{task_name}*.log",
    ]
    if repo:
        globs.extend(
            [
                f"{repo}/logs/{stem}_*.log",
                f"{repo}/logs/{parent_stem}_*.log",
                f"{repo}/logs/**/*{stem}*.log",
                f"{repo}/logs/**/*{parent_stem}*.log",
                f"{repo}/logs/**/*.log",
                f"{repo}/outputs/**/*.log",
                f"{repo}/outputs/**/log*.txt",
                f"{repo}/train_out/**/*.log",
                f"{repo}/wandb/**/logs/*.log",
            ]
        )
    return globs


def default_checkpoint_globs(script_rel: str) -> list[str]:
    script = Path(script_rel)
    repo = script.parts[0] if script.parts else ""
    if not repo:
        return []
    return [
        f"{repo}/outputs/**/checkpoints/**/*",
        f"{repo}/outputs/**/*ckpt*",
        f"{repo}/outputs/**/*.pt",
        f"{repo}/outputs/**/*.pth",
        f"{repo}/checkpoints/**/*",
        f"{repo}/ckpt/**/*",
        f"{repo}/train_out/**/*",
    ]


def inspect_script_capture(script_path: Path) -> dict[str, Any]:
    result = {
        "complete_terminal_capture": False,
        "has_top_level_tee": False,
        "has_torchrun_tee": False,
        "has_framework_logging": False,
        "notes": [],
    }
    if not script_path.exists():
        result["notes"].append("script_missing")
        return result
    try:
        text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result["notes"].append(f"script_unreadable: {exc}")
        return result

    result["has_top_level_tee"] = bool(CAPTURE_FULL_RE.search(text) and ("2>&1" in text or "2> >(" in text))
    result["has_torchrun_tee"] = "--tee" in text
    result["has_framework_logging"] = any(token in text for token in ("hydra.run.dir", "logging.mode", "WANDB_MODE"))
    result["complete_terminal_capture"] = result["has_top_level_tee"]

    if result["complete_terminal_capture"]:
        result["notes"].append("stdout_stderr_redirected_with_tee")
    elif result["has_torchrun_tee"]:
        result["notes"].append("torchrun_worker_tee_only; parent shell output may be incomplete")
    elif result["has_framework_logging"]:
        result["notes"].append("framework_logging_only; terminal output may be incomplete")
    else:
        result["notes"].append("no_obvious_terminal_log_capture")
    return result


def list_processes() -> list[dict[str, Any]]:
    commands = [
        ["ps", "-eo", "pid=,ppid=,etime=,args="],
        ["ps", "axo", "pid=,ppid=,etime=,command="],
    ]
    output = ""
    for cmd in commands:
        try:
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            break
        except Exception:
            continue
    processes: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, etime, command = parts
        if "monitor_training.py" in command:
            continue
        processes.append({"pid": int(pid), "ppid": int(ppid), "etime": etime, "command": command})
    return processes


def matching_processes(processes: list[dict[str, Any]], script_rel: str, script_abs: Path) -> list[dict[str, Any]]:
    script_name = Path(script_rel).name
    needles = {script_rel, str(script_abs), f"./{script_rel}", script_name}
    matches = []
    for proc in processes:
        command = proc["command"]
        if any(needle and needle in command for needle in needles):
            matches.append(proc)
    return matches


def extract_progress(text: str) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for line in text.splitlines()[-2000:]:
        if match := PROGRESS_RES[0].search(line):
            current = float(match.group(1)) * 1000.0
            total = float(match.group(2)) * 1000.0
            latest = {
                "type": "iteration",
                "current": int(current),
                "total": int(total),
                "percent": round(current / total * 100, 3) if total else None,
                "line": line[-500:],
            }
        elif match := PROGRESS_RES[1].search(line):
            latest = {"type": "step", "current": int(match.group(1)), "line": line[-500:]}
        elif match := PROGRESS_RES[2].search(line):
            latest = {"type": "step", "current": int(match.group(1)), "line": line[-500:]}
        elif match := PROGRESS_RES[3].search(line):
            item: dict[str, Any] = {"type": "epoch", "current": int(match.group(1)), "line": line[-500:]}
            if match.group(2):
                item["total"] = int(match.group(2))
                item["percent"] = round(item["current"] / item["total"] * 100, 3) if item["total"] else None
            latest = item
        elif match := PROGRESS_RES[4].search(line):
            latest = {
                "type": "tqdm",
                "current": int(match.group(2)),
                "total": int(match.group(3)),
                "percent": float(match.group(1)),
                "line": line[-500:],
            }
    return latest


def extract_error(text: str) -> dict[str, Any] | None:
    lines = text.splitlines()
    hit_idx = None
    for i, line in enumerate(lines):
        if ERROR_RE.search(line):
            hit_idx = i
    if hit_idx is None:
        return None
    start = max(0, hit_idx - 40)
    end = min(len(lines), hit_idx + 120)
    excerpt = "\n".join(lines[start:end])[-30_000:]
    return {
        "line": lines[hit_idx][-1000:],
        "excerpt": excerpt,
        "excerpt_line_count": end - start,
    }


def save_error_excerpt(output_path: Path, task_name: str, source: Path | None, error: dict[str, Any]) -> Path:
    errors_dir = output_path.parent / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    payload = f"{source or ''}\n{error.get('excerpt', '')}"
    digest = hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:12]
    path = errors_dir / f"{safe_name(task_name)}_{digest}.log"
    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            f.write(f"task: {task_name}\n")
            f.write(f"source: {source}\n")
            f.write(f"saved_at: {iso()}\n")
            f.write("=" * 80 + "\n")
            f.write(error.get("excerpt", ""))
            f.write("\n")
    return path


def path_info(path: Path | None, root: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    return {
        "path": rel,
        "absolute_path": str(path.resolve()),
        "modified_at": iso(stat.st_mtime),
        "age_seconds": round(time.time() - stat.st_mtime, 3),
        "size_bytes": stat.st_size if path.is_file() else None,
        "is_dir": path.is_dir(),
    }


def inspect_task(
    raw_task: Any,
    index: int,
    root: Path,
    output_path: Path,
    processes: list[dict[str, Any]],
    stale_after: int,
) -> dict[str, Any]:
    task = normalize_task(raw_task, index)
    script_rel = str(task.get("script", "")).strip()
    name = str(task.get("name") or Path(script_rel).stem or f"task_{index}")
    enabled = bool(task.get("enabled", True))
    script_abs = (root / script_rel).resolve() if script_rel else root
    exists = bool(script_rel and script_abs.exists())

    result: dict[str, Any] = {
        "name": name,
        "script": script_rel,
        "script_abs": str(script_abs),
        "enabled": enabled,
        "script_exists": exists,
        "status": "disabled" if not enabled else "unknown",
    }
    if "invalid" in task:
        result["status"] = "invalid_task"
        result["invalid"] = task["invalid"]
        return result
    if not enabled:
        return result
    if not script_rel:
        result["status"] = "missing_script_field"
        return result

    result["log_capture"] = inspect_script_capture(script_abs)
    procs = matching_processes(processes, script_rel, script_abs)
    result["running"] = bool(procs)
    result["processes"] = procs[:20]

    log_globs = list(task.get("log_globs", [])) or default_log_globs(script_rel, name)
    checkpoint_globs = list(task.get("checkpoint_globs", [])) or default_checkpoint_globs(script_rel)
    logs = glob_existing(root, log_globs)
    checkpoints = glob_existing(root, checkpoint_globs)
    latest_log = newest_path([p for p in logs if p.is_file()])
    latest_checkpoint = newest_path(checkpoints)
    result["latest_log"] = path_info(latest_log, root)
    result["latest_checkpoint"] = path_info(latest_checkpoint, root)
    result["matched_log_count"] = len(logs)
    result["matched_checkpoint_count"] = len(checkpoints)

    tail = read_tail(latest_log) if latest_log else ""
    result["progress"] = extract_progress(tail) if tail else None
    error = extract_error(tail) if tail else None
    if error:
        error_file = save_error_excerpt(output_path, name, latest_log, error)
        result["error"] = {
            "source": str(latest_log.relative_to(root)) if latest_log else None,
            "line": error["line"],
            "saved_to": str(error_file.relative_to(root)),
        }

    latest_log_age = result.get("latest_log", {}).get("age_seconds") if result.get("latest_log") else None
    if not exists:
        result["status"] = "missing_script"
    elif error and result["running"]:
        result["status"] = "running_with_error"
    elif error:
        result["status"] = "error"
    elif result["running"] and latest_log_age is not None and latest_log_age > stale_after:
        result["status"] = "running_stale"
    elif result["running"]:
        result["status"] = "running"
    elif result["latest_log"] or result["latest_checkpoint"]:
        result["status"] = "idle_or_completed"
    else:
        result["status"] = "idle_no_logs"

    return result


def build_state(root: Path, input_path: Path, output_path: Path, cli_interval: int | None) -> tuple[dict[str, Any], int]:
    config = read_json(input_path)
    interval = int(cli_interval or config.get("poll_interval_seconds", 60))
    stale_after = int(config.get("stale_after_seconds", 1800))
    processes = list_processes()
    tasks = [inspect_task(t, i, root, output_path, processes, stale_after) for i, t in enumerate(config.get("tasks", []))]
    summary = {
        "total": len(tasks),
        "enabled": sum(1 for t in tasks if t.get("enabled")),
        "running": sum(1 for t in tasks if t.get("running")),
        "errors": sum(1 for t in tasks if "error" in t),
        "missing_scripts": sum(1 for t in tasks if t.get("status") == "missing_script"),
        "incomplete_terminal_capture": sum(
            1
            for t in tasks
            if t.get("enabled") and not t.get("log_capture", {}).get("complete_terminal_capture", False)
        ),
    }
    state = {
        "generated_at": iso(),
        "project_root": str(root),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "poll_interval_seconds": interval,
        "summary": summary,
        "tasks": tasks,
    }
    return state, interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor training progress for umipolicy scripts.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root used to resolve relative task paths.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="JSON task list.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output state JSON path.")
    parser.add_argument("--interval", type=int, default=None, help="Override poll interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Run one poll and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    input_path = args.input if args.input.is_absolute() else root / args.input
    output_path = args.output if args.output.is_absolute() else root / args.output
    ensure_input(input_path)

    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    while True:
        try:
            state, interval = build_state(root, input_path, output_path, args.interval)
        except Exception as exc:  # Keep the monitor alive even if the input file is temporarily invalid.
            state = {
                "generated_at": iso(),
                "project_root": str(root),
                "input_path": str(input_path),
                "output_path": str(output_path),
                "summary": {"total": 0, "enabled": 0, "running": 0, "errors": 1},
                "monitor_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            interval = int(args.interval or 60)
        write_json_atomic(output_path, state)
        print(f"[{state['generated_at']}] wrote {output_path}")
        if args.once or stopping:
            break
        time.sleep(max(1, interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
