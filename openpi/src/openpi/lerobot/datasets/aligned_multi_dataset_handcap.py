#!/usr/bin/env python
"""Force/time aligned multi-dataset wrapper for Handcap LeRobot datasets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import hashlib
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset_handcap import LeRobotDatasetHandcap
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _stack_column(dataset, key: str) -> np.ndarray:
    values = dataset.hf_dataset[key]
    if isinstance(values, torch.Tensor):
        return _as_numpy(values)
    arrays = [_as_numpy(value) for value in values]
    try:
        return np.stack(arrays, axis=0)
    except ValueError:
        return np.asarray(arrays)


def _as_2d_feature(values: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[-1] < dim:
        pad_width = [(0, 0)] * values.ndim
        pad_width[-1] = (0, dim - values.shape[-1])
        values = np.pad(values, pad_width)
    return values[..., :dim].reshape(values.shape[0], dim)


def _extract_force(dataset: LeRobotDatasetHandcap, *, force_dim: int, action_base_dim: int) -> np.ndarray:
    column_names = set(dataset.hf_dataset.column_names)
    for key in ("observation.force", "observation.forces", "force"):
        if key in column_names:
            return _as_2d_feature(_stack_column(dataset, key), force_dim)

    left = None
    right = None
    for key in ("observation.forces.left", "observation.force_left"):
        if key in column_names:
            left = _as_2d_feature(_stack_column(dataset, key), 1)
            break
    for key in ("observation.forces.right", "observation.force_right"):
        if key in column_names:
            right = _as_2d_feature(_stack_column(dataset, key), 1)
            break
    if left is not None and right is not None:
        return _as_2d_feature(np.concatenate([left, right], axis=-1), force_dim)
    if left is not None:
        return _as_2d_feature(left, force_dim)

    if "observation.state" in column_names:
        state = _stack_column(dataset, "observation.state")
        state = np.asarray(state, dtype=np.float32)
        if state.ndim >= 2 and state.shape[-1] >= action_base_dim + force_dim:
            return _as_2d_feature(state[..., action_base_dim : action_base_dim + force_dim], force_dim)

    logger.warning("No force columns found for %s; using zeros for alignment.", dataset.repo_id)
    return np.zeros((len(dataset), force_dim), dtype=np.float32)


def _frame_progress(dataset: LeRobotDatasetHandcap) -> np.ndarray:
    episode_index = _stack_column(dataset, "episode_index").astype(np.int64).reshape(-1)
    if "timestamp" in dataset.hf_dataset.column_names:
        order_value = _stack_column(dataset, "timestamp").astype(np.float64).reshape(-1)
    elif "index" in dataset.hf_dataset.column_names:
        order_value = _stack_column(dataset, "index").astype(np.float64).reshape(-1)
    else:
        order_value = np.arange(len(dataset), dtype=np.float64)

    progress = np.zeros(len(dataset), dtype=np.float32)
    for ep_idx in np.unique(episode_index):
        indices = np.flatnonzero(episode_index == ep_idx)
        if len(indices) <= 1:
            progress[indices] = 0.0
            continue
        sorted_indices = indices[np.argsort(order_value[indices], kind="stable")]
        progress[sorted_indices] = np.linspace(0.0, 1.0, len(sorted_indices), dtype=np.float32)
    return progress


def _smooth_by_episode(dataset: LeRobotDatasetHandcap, force: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return force
    episode_index = _stack_column(dataset, "episode_index").astype(np.int64).reshape(-1)
    smoothed = force.copy()
    kernel = np.ones(window, dtype=np.float32) / float(window)
    for ep_idx in np.unique(episode_index):
        indices = np.flatnonzero(episode_index == ep_idx)
        if len(indices) <= 1:
            continue
        for dim in range(force.shape[-1]):
            smoothed[indices, dim] = np.convolve(force[indices, dim], kernel, mode="same")
    return smoothed


def _neighbor_bins(center: tuple[int, ...], radii: tuple[int, ...]):
    ranges = [range(value - radius, value + radius + 1) for value, radius in zip(center, radii, strict=True)]
    yield from itertools.product(*ranges)


class _AlignmentTable:
    def __init__(
        self,
        dataset: LeRobotDatasetHandcap,
        *,
        force_dim: int,
        action_base_dim: int,
        force_smoothing_window: int,
    ):
        force = _extract_force(dataset, force_dim=force_dim, action_base_dim=action_base_dim)
        self.force = _smooth_by_episode(dataset, force, force_smoothing_window)
        self.progress = _frame_progress(dataset)


class _GridMatcher:
    def __init__(
        self,
        table: _AlignmentTable,
        *,
        force_eps: float,
        max_progress_diff: float,
    ):
        self.table = table
        self.force_eps = max(float(force_eps), 1e-6)
        self.max_progress_diff = max(float(max_progress_diff), 1e-6)
        self.force_bin_size = self.force_eps
        self.progress_bin_size = self.max_progress_diff
        self._buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)

        force_bins = np.floor(table.force / self.force_bin_size).astype(np.int64)
        progress_bins = np.floor(table.progress / self.progress_bin_size).astype(np.int64)
        for idx, force_bin in enumerate(force_bins):
            key = tuple(force_bin.tolist()) + (int(progress_bins[idx]),)
            self._buckets[key].append(idx)

    def find(
        self,
        force: np.ndarray,
        progress: float,
        *,
        time_weight: float,
    ) -> int:
        if not np.all(np.isfinite(force)):
            return -1

        force_bin = tuple(np.floor(force / self.force_bin_size).astype(np.int64).tolist())
        progress_bin = int(np.floor(progress / self.progress_bin_size))
        center = force_bin + (progress_bin,)
        radii = (1,) * len(force_bin) + (1,)

        best_idx = -1
        best_cost = np.inf
        for key in _neighbor_bins(center, radii):
            candidates = self._buckets.get(key)
            if not candidates:
                continue
            candidate_indices = np.asarray(candidates, dtype=np.int64)
            progress_dist = np.abs(self.table.progress[candidate_indices] - progress)
            time_ok = progress_dist <= self.max_progress_diff
            if not np.any(time_ok):
                continue

            candidate_indices = candidate_indices[time_ok]
            progress_dist = progress_dist[time_ok]
            force_dist = np.linalg.norm(self.table.force[candidate_indices] - force[None, :], axis=-1)
            force_ok = force_dist <= self.force_eps
            if not np.any(force_ok):
                continue

            candidate_indices = candidate_indices[force_ok]
            progress_dist = progress_dist[force_ok]
            force_dist = force_dist[force_ok]
            costs = force_dist / self.force_eps + float(time_weight) * progress_dist
            local_best = int(np.argmin(costs))
            if costs[local_best] < best_cost:
                best_cost = float(costs[local_best])
                best_idx = int(candidate_indices[local_best])
        return best_idx


class AlignedMultiLeRobotDatasetHandcap(torch.utils.data.Dataset):
    """Returns force/time aligned samples from several Handcap datasets.

    Indexing is flattened as ``alignment_id * num_datasets + health_id`` so a
    grouped batch sampler can keep all health variants of each alignment together.
    """

    def __init__(
        self,
        repo_ids: Sequence[str],
        roots: Sequence[str | Path],
        *,
        health_labels: Sequence[str] | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        force_dim: int = 2,
        action_base_dim: int = 10,
        force_eps: float = 0.5,
        max_progress_diff: float = 0.15,
        time_weight: float = 0.25,
        force_smoothing_window: int = 3,
        anchor_dataset_index: int = 1,
        max_alignments: int | None = None,
        seed: int = 0,
        cache_dir: str | Path | None = None,
        rebuild_cache: bool = False,
    ):
        super().__init__()
        if len(repo_ids) != len(roots):
            raise ValueError("repo_ids and roots must have the same length.")
        if len(repo_ids) < 2:
            raise ValueError("At least two datasets are required for alignment.")
        if health_labels is not None and len(health_labels) != len(repo_ids):
            raise ValueError("health_labels must match repo_ids length.")
        if not 0 <= anchor_dataset_index < len(repo_ids):
            raise ValueError("anchor_dataset_index is out of range.")

        self.repo_ids = tuple(repo_ids)
        self.roots = tuple(
            Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
            for repo_id, root in zip(repo_ids, roots, strict=True)
        )
        self.health_labels = tuple(health_labels or [str(i) for i in range(len(repo_ids))])
        self.group_size = len(self.repo_ids)
        self.anchor_dataset_index = int(anchor_dataset_index)
        self.force_eps = float(force_eps)
        self.max_progress_diff = float(max_progress_diff)
        self.time_weight = float(time_weight)
        self.force_smoothing_window = int(force_smoothing_window)
        self.max_alignments = max_alignments
        self.seed = int(seed)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(self.repo_ids, 0.0001)
        self._datasets = [
            LeRobotDatasetHandcap(
                repo_id,
                root=root,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id, root in zip(self.repo_ids, self.roots, strict=True)
        ]
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])
        self.alignment_indices = self._load_or_build_alignment_indices(
            force_dim=force_dim,
            action_base_dim=action_base_dim,
            rebuild_cache=rebuild_cache,
        )
        if self.alignment_indices.size == 0:
            raise RuntimeError(
                "No aligned frames were found. Relax force_eps/max_progress_diff or check force columns."
            )
        logger.info(
            "Built aligned Handcap dataset with %d groups across %d datasets.",
            self.num_alignments,
            self.group_size,
        )

    @property
    def num_alignments(self) -> int:
        return int(self.alignment_indices.shape[0])

    @property
    def fps(self) -> int:
        return self._datasets[0].fps

    @property
    def meta(self):
        return self._datasets[0].meta

    def __len__(self) -> int:
        return self.num_alignments * self.group_size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        alignment_id, health_id = divmod(int(idx), self.group_size)
        source_idx = int(self.alignment_indices[alignment_id, health_id])
        item = self._datasets[health_id][source_idx]
        item["alignment_id"] = torch.tensor(alignment_id, dtype=torch.int64)
        item["health_id"] = torch.tensor(health_id, dtype=torch.int64)
        item["aligned_source_index"] = torch.tensor(source_idx, dtype=torch.int64)
        return item

    def _cache_path(self, *, force_dim: int, action_base_dim: int) -> Path | None:
        if self.cache_dir is None:
            return None
        payload = {
            "repo_ids": self.repo_ids,
            "roots": [str(root.resolve()) for root in self.roots],
            "health_labels": self.health_labels,
            "frames": [len(dataset) for dataset in self._datasets],
            "force_dim": force_dim,
            "action_base_dim": action_base_dim,
            "force_eps": self.force_eps,
            "max_progress_diff": self.max_progress_diff,
            "time_weight": self.time_weight,
            "force_smoothing_window": self.force_smoothing_window,
            "anchor_dataset_index": self.anchor_dataset_index,
            "max_alignments": self.max_alignments,
        }
        key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"aligned_handcap_{key}.npz"

    def _load_or_build_alignment_indices(
        self,
        *,
        force_dim: int,
        action_base_dim: int,
        rebuild_cache: bool,
    ) -> np.ndarray:
        cache_path = self._cache_path(force_dim=force_dim, action_base_dim=action_base_dim)
        if cache_path is not None and cache_path.exists() and not rebuild_cache:
            with np.load(cache_path) as data:
                alignment_indices = data["alignment_indices"].astype(np.int64)
            if alignment_indices.ndim == 2 and alignment_indices.shape[1] == self.group_size:
                return alignment_indices

        alignment_indices = self._build_alignment_indices(force_dim=force_dim, action_base_dim=action_base_dim)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, alignment_indices=alignment_indices)
        return alignment_indices

    def _build_alignment_indices(self, *, force_dim: int, action_base_dim: int) -> np.ndarray:
        tables = [
            _AlignmentTable(
                dataset,
                force_dim=force_dim,
                action_base_dim=action_base_dim,
                force_smoothing_window=self.force_smoothing_window,
            )
            for dataset in self._datasets
        ]
        matchers = [
            None
            if idx == self.anchor_dataset_index
            else _GridMatcher(table, force_eps=self.force_eps, max_progress_diff=self.max_progress_diff)
            for idx, table in enumerate(tables)
        ]

        anchor_table = tables[self.anchor_dataset_index]
        rows = []
        for anchor_idx in range(anchor_table.force.shape[0]):
            row = np.full((self.group_size,), -1, dtype=np.int64)
            row[self.anchor_dataset_index] = anchor_idx
            valid = True
            for dataset_idx, matcher in enumerate(matchers):
                if matcher is None:
                    continue
                match_idx = matcher.find(
                    anchor_table.force[anchor_idx],
                    float(anchor_table.progress[anchor_idx]),
                    time_weight=self.time_weight,
                )
                if match_idx < 0:
                    valid = False
                    break
                row[dataset_idx] = match_idx
            if valid:
                rows.append(row)

        if not rows:
            return np.empty((0, self.group_size), dtype=np.int64)

        alignment_indices = np.stack(rows, axis=0)
        if (
            self.max_alignments is not None
            and self.max_alignments > 0
            and len(alignment_indices) > self.max_alignments
        ):
            rng = np.random.default_rng(self.seed)
            selected = np.sort(rng.choice(len(alignment_indices), size=self.max_alignments, replace=False))
            alignment_indices = alignment_indices[selected]
        return alignment_indices
