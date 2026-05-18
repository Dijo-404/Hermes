#!/usr/bin/env python3
"""
train.py — train PSIPredictor on Phase-2 labeled CSVs with LOSO eval.

Purpose
-------
Trains the LSTM defined in `model.py` against a labeled CSV emitted by
`label.py`. Evaluates with leave-one-scenario-out cross-validation
(plan-executable.md Phase 3 task 3) and reports the target metrics:

    - precision (≥ 0.70)
    - recall    (≥ 0.85)
    - lead-time histogram: ≥ 80 % of true positives must fire ≥ 100 ms
      before the actual kill instant.

Loss: BCEWithLogitsLoss(pos_weight=10.0). We drop the sigmoid in
the training forward pass for numerical stability, then put it back for
inference & ONNX export by switching to `model(x)` (the default forward
applies sigmoid). See model.py docstring.

Outputs
-------
Under `--out-dir`:

    psi_predictor.pt          state_dict + arch hyperparameters
    normalization.json        z-score statistics fit on full train data
    loso_metrics.csv          per-fold precision / recall
    lead_time_histogram.csv   ms-before-kill bucket counts
    model_card.md             filled-in template from model_card.md skeleton

Usage
-----
    python train.py --data data/all.labeled.csv --out-dir out/ \\
                    --epochs 30 --batch 256 --lr 1e-3 --seed 42 \\
                    [--device cuda] [--quick]

`--quick`: 2 epochs, only the first two scenarios as folds, used for
plan-executable.md Phase 3 verification (smoke test of the pipeline).

Exit codes
----------
  0 — training completed; all artifacts written; param-count assertion held.
  1 — CSV missing / unreadable / schema mismatch.
  2 — fewer than 2 scenarios in CSV (LOSO needs at least 2).
  3 — parameter-count assertion failed (>200,000).
  4 — final-model save / sidecar write failed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    FEATURES,
    WINDOW_LEN,
    NormStats,
    PSISeriesDataset,
    fit_stats,
)
from model import PSIPredictor, count_parameters


PARAM_BUDGET: int = 200_000
KILL_THRESHOLD: float = 0.65          # plan §Phase 4 default
LEAD_TIME_TARGET_MS: int = 100
LEAD_TIME_TARGET_FRAC: float = 0.80
SAMPLE_PERIOD_MS: int = 100           # collector cadence


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PSIPredictor with LOSO CV.")
    p.add_argument("--data", required=True, type=Path, help="Labeled CSV.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 2 epochs on first two scenarios only.",
    )
    p.add_argument(
        "--pos-weight",
        type=float,
        default=10.0,
        help="Positive-class weight for BCEWithLogitsLoss (plan §Phase 3).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=KILL_THRESHOLD,
        help="Decision threshold for precision/recall + lead-time analysis.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    held_out: str
    precision: float
    recall: float
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int


def precision_recall(
    probs: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float, int, int, int]:
    preds = (probs >= threshold).astype(np.int32)
    targets = (labels >= 0.5).astype(np.int32)
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, tp, fp, fn


def lead_time_ms_for_true_positives(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    sample_period_ms: int = SAMPLE_PERIOD_MS,
) -> np.ndarray:
    """
    For each contiguous run of positive labels (corresponding to a single
    kill's pre-kill window), find the FIRST index where probs >= threshold
    (a true positive). Lead time in ms is:

        (end_of_positive_run_index - first_alarm_index) * sample_period_ms
        + 100  (because label.py centres the window at T_k - 200..-100ms)

    The +100 ms reflects that even the *last* labelled-positive row sits
    100 ms before the kill instant (label.py defaults: lead=200, window=100).
    Returns ms before kill for each detected positive run; runs we missed
    contribute nothing (they're counted as false negatives elsewhere).
    """
    leads: list[float] = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] < 0.5:
            i += 1
            continue
        # Found a positive run; find its bounds.
        j = i
        while j < n and labels[j] >= 0.5:
            j += 1
        # Positive run spans [i, j).
        # First alarm within this run:
        run_probs = probs[i:j]
        alarm_offsets = np.where(run_probs >= threshold)[0]
        if alarm_offsets.size > 0:
            first_alarm_idx = int(alarm_offsets[0])  # offset within run
            # The LAST row of the run is at index (j - 1), which sits
            # ~100 ms before T_k. The alarm at i+first_alarm_idx sits
            # ((j - 1) - (i + first_alarm_idx)) * 100ms + 100ms before T_k.
            lead_ms = (j - 1 - (i + first_alarm_idx)) * sample_period_ms + 100
            leads.append(float(lead_ms))
        i = j
    return np.asarray(leads, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------

def train_one(
    model: PSIPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(windows, return_logits=True)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * windows.size(0)
        n += windows.size(0)
    return total / max(n, 1)


@torch.no_grad()
def collect_probs(
    model: PSIPredictor, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        probs = model(windows)  # sigmoid-applied path
        probs_chunks.append(probs.detach().cpu().numpy().reshape(-1))
        labels_chunks.append(labels.numpy().reshape(-1))
    if not probs_chunks:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    return (
        np.concatenate(probs_chunks).astype(np.float32),
        np.concatenate(labels_chunks).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# LOSO fold runner.
# ---------------------------------------------------------------------------

def run_fold(
    csv_path: Path,
    train_scenarios: list[str],
    val_scenario: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FoldResult, np.ndarray]:
    """Train on `train_scenarios`, evaluate on `val_scenario`. Returns
    (fold metrics, lead_time_ms_array)."""
    train_ds = PSISeriesDataset(csv_path, scenarios=train_scenarios)
    val_ds = PSISeriesDataset(csv_path, scenarios=[val_scenario])

    stats = fit_stats(train_ds.raw_features())
    train_ds.apply_stats(stats)
    val_ds.apply_stats(stats)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, drop_last=False
    )

    model = PSIPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = 2 if args.quick else args.epochs
    for epoch in range(epochs):
        train_loss = train_one(model, train_loader, optimizer, loss_fn, device)
        print(
            f"  [fold={val_scenario}] epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f}"
        )

    probs, labels = collect_probs(model, val_loader, device)
    p, r, tp, fp, fn = precision_recall(probs, labels, args.threshold)
    leads = lead_time_ms_for_true_positives(probs, labels, args.threshold)

    return (
        FoldResult(
            held_out=val_scenario,
            precision=p,
            recall=r,
            n_true_positives=tp,
            n_false_positives=fp,
            n_false_negatives=fn,
        ),
        leads,
    )


# ---------------------------------------------------------------------------
# Final-model training on the full dataset.
# ---------------------------------------------------------------------------

def train_final(
    csv_path: Path, scenarios: list[str], args: argparse.Namespace, device: torch.device
) -> tuple[PSIPredictor, NormStats]:
    ds = PSISeriesDataset(csv_path, scenarios=scenarios)
    stats = fit_stats(ds.raw_features())
    ds.apply_stats(stats)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)
    model = PSIPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    epochs = 2 if args.quick else args.epochs
    print(f"[final] training on {len(scenarios)} scenarios, {len(ds)} windows")
    for epoch in range(epochs):
        loss = train_one(model, loader, optimizer, loss_fn, device)
        print(f"  [final] epoch {epoch + 1}/{epochs} train_loss={loss:.4f}")
    return model, stats


# ---------------------------------------------------------------------------
# Reporting helpers.
# ---------------------------------------------------------------------------

def write_loso_metrics(out_path: Path, folds: list[FoldResult]) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["held_out", "precision", "recall", "tp", "fp", "fn"]
        )
        for fr in folds:
            w.writerow(
                [
                    fr.held_out,
                    f"{fr.precision:.6f}",
                    f"{fr.recall:.6f}",
                    fr.n_true_positives,
                    fr.n_false_positives,
                    fr.n_false_negatives,
                ]
            )


def write_lead_histogram(out_path: Path, leads_ms: np.ndarray) -> None:
    # Buckets: <0, 0-100, 100-200, 200-300, 300-500, 500-1000, >=1000 ms.
    edges = [-float("inf"), 0, 100, 200, 300, 500, 1000, float("inf")]
    bucket_labels = [
        "<0ms",
        "0-100ms",
        "100-200ms",
        "200-300ms",
        "300-500ms",
        "500-1000ms",
        ">=1000ms",
    ]
    counts = [0] * (len(edges) - 1)
    for v in leads_ms:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                counts[i] += 1
                break
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "count"])
        for lbl, c in zip(bucket_labels, counts):
            w.writerow([lbl, c])


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def fill_model_card(
    template_path: Path,
    out_path: Path,
    *,
    param_count: int,
    data_sha256: str,
    git_sha: str,
    folds: list[FoldResult],
    lead_histogram_path: Path,
    lead_target_frac_actual: float,
) -> None:
    if not template_path.exists():
        # Don't fail training if the template went missing — emit a stub.
        out_path.write_text(
            f"# psi_predictor model card\n\n"
            f"param_count: {param_count}\ndata_sha256: {data_sha256}\n"
            f"git_commit: {git_sha}\n"
        )
        return
    body = template_path.read_text()
    mean_p = float(np.mean([f.precision for f in folds])) if folds else 0.0
    std_p = float(np.std([f.precision for f in folds])) if folds else 0.0
    mean_r = float(np.mean([f.recall for f in folds])) if folds else 0.0
    std_r = float(np.std([f.recall for f in folds])) if folds else 0.0
    fold_lines = "\n".join(
        f"| {f.held_out} | {f.precision:.3f} | {f.recall:.3f} | "
        f"{f.n_true_positives} | {f.n_false_positives} | {f.n_false_negatives} |"
        for f in folds
    )
    filled = (
        body.replace("{{PARAM_COUNT}}", str(param_count))
        .replace("{{DATA_SHA256}}", data_sha256)
        .replace("{{GIT_COMMIT}}", git_sha)
        .replace("{{LOSO_FOLDS_TABLE}}", fold_lines)
        .replace("{{LOSO_MEAN_PRECISION}}", f"{mean_p:.3f} ± {std_p:.3f}")
        .replace("{{LOSO_MEAN_RECALL}}", f"{mean_r:.3f} ± {std_r:.3f}")
        .replace("{{LEAD_HISTOGRAM_PATH}}", str(lead_histogram_path))
        .replace("{{LEAD_GE_100MS_FRAC}}", f"{lead_target_frac_actual:.3f}")
        .replace("{{ONNX_OPSET}}", "11")
    )
    out_path.write_text(filled)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    seed_all(args.seed)

    if not args.data.exists():
        print(f"ERR: data CSV not found: {args.data}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Discover scenarios by reading once with no filter.
    probe = PSISeriesDataset(args.data, scenarios=None)
    all_scenarios = probe.scenarios_seen()
    if len(all_scenarios) < 2:
        print(
            f"ERR: need ≥ 2 scenarios for LOSO; found {all_scenarios}",
            file=sys.stderr,
        )
        return 2

    if args.quick:
        all_scenarios = all_scenarios[:2]
        print(f"[quick] restricting to scenarios: {all_scenarios}")

    print(f"Discovered scenarios: {all_scenarios}")

    folds: list[FoldResult] = []
    all_leads: list[np.ndarray] = []

    for val_scen in all_scenarios:
        train_scens = [s for s in all_scenarios if s != val_scen]
        print(f"\n=== LOSO fold: held out = {val_scen} ===")
        fold, leads = run_fold(args.data, train_scens, val_scen, args, device)
        folds.append(fold)
        all_leads.append(leads)
        print(
            f"  -> precision={fold.precision:.3f} recall={fold.recall:.3f} "
            f"(tp={fold.n_true_positives} fp={fold.n_false_positives} "
            f"fn={fold.n_false_negatives})"
        )

    # Aggregate metrics.
    p_mean = float(np.mean([f.precision for f in folds]))
    p_std = float(np.std([f.precision for f in folds]))
    r_mean = float(np.mean([f.recall for f in folds]))
    r_std = float(np.std([f.recall for f in folds]))
    print(
        f"\nLOSO mean precision={p_mean:.3f} ± {p_std:.3f}, "
        f"recall={r_mean:.3f} ± {r_std:.3f}"
    )

    leads_concat = (
        np.concatenate(all_leads) if any(len(x) for x in all_leads)
        else np.zeros(0, dtype=np.float32)
    )
    ge_100 = (
        float((leads_concat >= LEAD_TIME_TARGET_MS).mean())
        if leads_concat.size > 0
        else 0.0
    )
    print(
        f"Lead-time ≥ {LEAD_TIME_TARGET_MS}ms fraction: {ge_100:.3f} "
        f"(target ≥ {LEAD_TIME_TARGET_FRAC})"
    )

    # Write fold + histogram artifacts.
    write_loso_metrics(args.out_dir / "loso_metrics.csv", folds)
    write_lead_histogram(args.out_dir / "lead_time_histogram.csv", leads_concat)

    # Train final model on all scenarios for export.
    print("\n=== Final training on full dataset ===")
    final_model, final_stats = train_final(
        args.data, all_scenarios, args, device
    )

    # Parameter count assertion.
    pc = count_parameters(final_model)
    print(f"\nParameter count: {pc}")
    if pc > PARAM_BUDGET:
        print(
            f"ERR: parameter count {pc} > budget {PARAM_BUDGET}",
            file=sys.stderr,
        )
        return 3
    assert pc <= PARAM_BUDGET, f"param count {pc} > {PARAM_BUDGET}"

    # Save artifacts.
    try:
        ckpt_path = args.out_dir / "psi_predictor.pt"
        torch.save(
            {
                "state_dict": final_model.state_dict(),
                "arch": {
                    "input_size": final_model.input_size,
                    "hidden_size": final_model.hidden_size,
                    "num_layers": final_model.num_layers,
                    "window_len": WINDOW_LEN,
                    "feature_order": FEATURES,
                },
                "param_count": pc,
            },
            ckpt_path,
        )
        final_stats.to_json(args.out_dir / "normalization.json")
    except OSError as e:
        print(f"ERR: could not save artifacts: {e}", file=sys.stderr)
        return 4

    # Model card.
    data_hash = sha256_of_file(args.data)
    fill_model_card(
        template_path=Path(__file__).parent / "model_card.md",
        out_path=args.out_dir / "model_card.md",
        param_count=pc,
        data_sha256=data_hash,
        git_sha=git_commit(),
        folds=folds,
        lead_histogram_path=args.out_dir / "lead_time_histogram.csv",
        lead_target_frac_actual=ge_100,
    )

    print(f"\nArtifacts written under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
