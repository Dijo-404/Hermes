#!/usr/bin/env python3
"""
dataset.py — rolling-window Dataset over Phase-2 labeled CSVs.

Purpose
-------
Wraps a labeled CSV (produced by `collector.py` + `label.py`) as a
`torch.utils.data.Dataset` that yields fixed-length rolling windows of
PSI / meminfo features alongside the kill_event label of the LAST row of
each window. This matches the C++ inference contract in Phase 4: the
predictor sees the most recent 20 samples (≈ 2 s of history at 100 ms
cadence) and emits one probability that the *next* tick will be a kill.

Feature order (plan-executable.md Phase 3 task 1) — DO NOT REORDER
-----------------------------------------------------------------
    1. some_avg10
    2. some_avg60
    3. some_total
    4. full_avg10
    5. full_total
    6. mem_available_kb

Window construction rules
-------------------------
- Window length is 20 samples.
- Windows are built within each (scenario, contiguous-time) group.
- A "contiguous" run is broken whenever two consecutive `timestamp_unix`
  values are more than `max_gap_ms` apart (default 200 ms; the collector
  samples at 100 ms, so a 200 ms gap means at least one sample was lost).
  Windows do not cross such breaks, nor do they cross scenario boundaries.
- Label = `kill_event` value of the LAST row in the window.

Normalization
-------------
Z-score per feature, statistics computed on the training-split rows
*before* windowing (so the stats reflect the underlying sample distribution,
not the heavily-overlapping window distribution).

The fitted statistics are serialised to a JSON sidecar so the C++
inference path (Phase 4) can apply byte-identical normalization:

    {
        "feature_order": [...],
        "mean":          [...],
        "std":           [...]
    }

Usage
-----
    from research.dataset import PSISeriesDataset, FEATURES, fit_stats

    ds_train = PSISeriesDataset(csv_path, scenarios=["idle","web"])
    stats    = fit_stats(ds_train.raw_features())
    ds_train.apply_stats(stats)
    ds_val   = PSISeriesDataset(csv_path, scenarios=["game"])
    ds_val.apply_stats(stats)

Exit codes
----------
N/A — library module.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants — feature order is load-bearing; the C++ side mirrors this.
# ---------------------------------------------------------------------------

FEATURES: list[str] = [
    "some_avg10",
    "some_avg60",
    "some_total",
    "full_avg10",
    "full_total",
    "mem_available_kb",
]

WINDOW_LEN: int = 20
DEFAULT_MAX_GAP_MS: int = 200


# ---------------------------------------------------------------------------
# Normalization stats container.
# ---------------------------------------------------------------------------

@dataclass
class NormStats:
    """Per-feature z-score statistics."""

    feature_order: list[str]
    mean: np.ndarray  # shape [F]
    std: np.ndarray   # shape [F]

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "feature_order": list(self.feature_order),
                    "mean": [float(x) for x in self.mean.tolist()],
                    "std": [float(x) for x in self.std.tolist()],
                },
                indent=2,
            )
        )

    @classmethod
    def from_json(cls, path: Path) -> "NormStats":
        obj = json.loads(Path(path).read_text())
        return cls(
            feature_order=list(obj["feature_order"]),
            mean=np.asarray(obj["mean"], dtype=np.float32),
            std=np.asarray(obj["std"], dtype=np.float32),
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply z-score normalization. x shape: [..., F]."""
        # Guard against zero variance.
        safe_std = np.where(self.std == 0.0, 1.0, self.std)
        return (x.astype(np.float32) - self.mean) / safe_std


def fit_stats(
    rows: np.ndarray, feature_order: Sequence[str] = FEATURES
) -> NormStats:
    """
    Fit z-score statistics over a [N, F] array of raw feature rows.
    Use this on the TRAIN split only; pass the result to val/test splits.
    """
    if rows.ndim != 2 or rows.shape[1] != len(feature_order):
        raise ValueError(
            f"rows must be [N, {len(feature_order)}], got {rows.shape}"
        )
    mean = rows.mean(axis=0).astype(np.float32)
    std = rows.std(axis=0).astype(np.float32)
    return NormStats(
        feature_order=list(feature_order), mean=mean, std=std
    )


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------

class PSISeriesDataset(Dataset):
    """
    Rolling-window dataset over a Phase-2 labeled CSV.

    Each sample is (window, label):
      window: FloatTensor[20, 6]  (normalized if apply_stats() was called)
      label:  FloatTensor[1]      (kill_event of last row in the window)
    """

    def __init__(
        self,
        csv_path: Path | str,
        scenarios: Iterable[str] | None = None,
        feature_order: Sequence[str] = FEATURES,
        window_len: int = WINDOW_LEN,
        max_gap_ms: int = DEFAULT_MAX_GAP_MS,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.feature_order = list(feature_order)
        self.window_len = window_len
        self.max_gap_s = max_gap_ms / 1000.0
        self._stats: NormStats | None = None

        scenarios_set = set(scenarios) if scenarios is not None else None

        # Read rows in CSV order, grouped by (scenario, contiguous-time).
        self._features_raw: np.ndarray  # [N_total_rows, F]
        self._labels_raw: np.ndarray    # [N_total_rows]
        self._window_start_idx: list[int]  # indices where windows may start
        self._scenarios_seen: list[str] = []

        (
            self._features_raw,
            self._labels_raw,
            self._window_start_idx,
            self._scenarios_seen,
        ) = self._load_and_index(self.csv_path, scenarios_set)

    # -- public API ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._window_start_idx)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self._window_start_idx[idx]
        end = start + self.window_len  # exclusive
        window = self._features_raw[start:end]
        if self._stats is not None:
            window = self._stats.transform(window)
        label = self._labels_raw[end - 1]
        return (
            torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)),
            torch.tensor([label], dtype=torch.float32),
        )

    def raw_features(self) -> np.ndarray:
        """All raw (unnormalized) feature rows. Use to fit stats."""
        return self._features_raw

    def labels_per_window(self) -> np.ndarray:
        """The label of the final row of every emitted window."""
        return np.asarray(
            [self._labels_raw[s + self.window_len - 1] for s in self._window_start_idx],
            dtype=np.float32,
        )

    def apply_stats(self, stats: NormStats) -> None:
        """Attach a fitted NormStats; subsequent __getitem__ will use it."""
        if list(stats.feature_order) != list(self.feature_order):
            raise ValueError(
                "Feature order mismatch: "
                f"stats={stats.feature_order} vs ds={self.feature_order}"
            )
        self._stats = stats

    def scenarios_seen(self) -> list[str]:
        """All distinct scenario tags encountered in the CSV (in first-seen order)."""
        return list(self._scenarios_seen)

    # -- internal -----------------------------------------------------------

    def _load_and_index(
        self,
        path: Path,
        scenarios_filter: set[str] | None,
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
        feats: list[list[float]] = []
        labels: list[float] = []
        # group boundaries: list of (start_row, end_row_exclusive) over `feats`.
        groups: list[tuple[int, int]] = []
        seen_order: list[str] = []
        seen_set: set[str] = set()

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for col in (*self.feature_order, "timestamp_unix", "scenario", "kill_event"):
                if col not in reader.fieldnames:  # type: ignore[union-attr]
                    raise ValueError(
                        f"CSV missing required column '{col}'. "
                        f"Have: {reader.fieldnames}"
                    )

            current_scenario: str | None = None
            current_group_start: int = 0
            last_ts: float | None = None

            for row in reader:
                scen = row["scenario"]
                if scenarios_filter is not None and scen not in scenarios_filter:
                    # Force a group break on filtered rows by resetting state.
                    if current_scenario is not None and len(feats) > current_group_start:
                        groups.append((current_group_start, len(feats)))
                    current_scenario = None
                    last_ts = None
                    current_group_start = len(feats)
                    continue

                # Parse timestamp + feature row; tolerate empty cells (collector
                # writes "" on a parse miss — those rows are unusable).
                try:
                    ts = float(row["timestamp_unix"])
                    feat_vals = [float(row[c]) for c in self.feature_order]
                    lbl = float(row["kill_event"])
                except (ValueError, TypeError):
                    # Missing/garbage — break the contiguous run.
                    if current_scenario is not None and len(feats) > current_group_start:
                        groups.append((current_group_start, len(feats)))
                    current_scenario = None
                    last_ts = None
                    current_group_start = len(feats)
                    continue

                # Detect group break: scenario change OR timestamp gap.
                gap_break = (
                    last_ts is not None and (ts - last_ts) > self.max_gap_s
                )
                scen_break = scen != current_scenario

                if scen_break or gap_break:
                    if current_scenario is not None and len(feats) > current_group_start:
                        groups.append((current_group_start, len(feats)))
                    current_group_start = len(feats)
                    current_scenario = scen
                    if scen not in seen_set:
                        seen_set.add(scen)
                        seen_order.append(scen)

                feats.append(feat_vals)
                labels.append(lbl)
                last_ts = ts

            # Flush trailing group.
            if current_scenario is not None and len(feats) > current_group_start:
                groups.append((current_group_start, len(feats)))

        if not feats:
            raise ValueError(
                f"No usable rows in {path} for scenarios={scenarios_filter}"
            )

        feats_arr = np.asarray(feats, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=np.float32)

        # Enumerate every valid window start within each contiguous group.
        starts: list[int] = []
        for gs, ge in groups:
            # Need WINDOW_LEN rows; last valid start is ge - WINDOW_LEN.
            last_start = ge - self.window_len
            if last_start < gs:
                continue
            starts.extend(range(gs, last_start + 1))

        return feats_arr, labels_arr, starts, seen_order
