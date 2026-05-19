#!/usr/bin/env python3
"""
aggregate.py — Phase 5 raw-artifact aggregator.

Purpose
-------
Walks a directory of per-cell run outputs produced by ab.sh +
collect_metrics.sh and reduces each cell to a single CSV row. The
analysis script (analyze.py) consumes the resulting CSV; this script
performs no statistics.

Per-cell inputs (under <run-dir>/):
  - manifest.json
  - lmkd.log                   # logcat -s lmkd:I lmkd-ml:I
  - gfxinfo_<pkg>_NNNN.txt     # dumpsys gfxinfo snapshots
  - lmkd_status_NNNN.txt       # /proc/<lmkd-pid>/status snapshots
  - coldstart_pre.txt
  - coldstart_post.txt
  - meta.txt

Output columns:
  workload, ml, run, jank_pct, kills_per_hour,
  inf_p50_ms, inf_p99_ms, vmrss_delta_kb, coldstart_p50_ms

Usage
-----
    python aggregate.py --in '<run-dirs-glob>' --out <summary-csv>

The --in glob is expanded by Python (not the shell); quote it.
Example: --in './results/*'

Exit codes
----------
  0 aggregation completed; CSV written.
  1 CLI / argument validation failure.
  2 No run directories matched the glob.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# ---- parse helpers --------------------------------------------------------

# `Kill 'name' (pid), uid X, oom_score_adj Y to free ...` from lmkd.cpp 2506.
_RE_KILL = re.compile(r"\bKill\s+'[^']+'\s*\(\d+\)")

# `inference latency p50=NNN us p99=MMM us (n=...)` from ml_predictor.cpp:251.
_RE_INF = re.compile(
    r"inference latency p50=(\d+)\s*us\s+p99=(\d+)\s*us"
)

# `Janky frames: N (P%)` from `dumpsys gfxinfo`. Modern Android also emits
# `Number Janky frames: N` then `Janky frames: P %` — match both.
_RE_JANKY_FRAMES_PCT = re.compile(
    r"(?:Janky\s+frames|Janky\s+frames\s*\(legacy\))\s*:\s*\d+\s*\(\s*([\d.]+)\s*%\s*\)",
    re.IGNORECASE,
)

# `VmRSS:   12345 kB` from /proc/<pid>/status.
_RE_VMRSS = re.compile(r"^VmRSS:\s+(\d+)\s+kB", re.MULTILINE)

# `am start -W` block: TotalTime: N (ms).
_RE_TOTALTIME = re.compile(r"^\s*TotalTime:\s*(\d+)", re.MULTILINE)


def parse_kills_per_hour(lmkd_log: Path, duration_sec: float) -> float:
    if not lmkd_log.exists() or duration_sec <= 0:
        return 0.0
    with lmkd_log.open("r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    kills = len(_RE_KILL.findall(text))
    return kills * 3600.0 / duration_sec


def parse_inference_latency(lmkd_log: Path) -> tuple[Optional[float], Optional[float]]:
    """Return (p50_ms, p99_ms) as medians across the rolling-window log lines.

    ml_predictor.cpp emits a fresh p50/p99 line every 10 seconds; each line
    already summarises a rolling window. The median across emitted lines is
    a robust per-run point estimate.
    """
    if not lmkd_log.exists():
        return (None, None)
    p50s_us: list[int] = []
    p99s_us: list[int] = []
    with lmkd_log.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _RE_INF.search(line)
            if m:
                p50s_us.append(int(m.group(1)))
                p99s_us.append(int(m.group(2)))
    if not p50s_us:
        return (None, None)
    return (
        statistics.median(p50s_us) / 1000.0,
        statistics.median(p99s_us) / 1000.0,
    )


def parse_jank_pct(run_dir: Path) -> Optional[float]:
    """Average janky-frames percent across all gfxinfo snapshots in the cell."""
    pcts: list[float] = []
    for snap in sorted(run_dir.glob("gfxinfo_*.txt")):
        try:
            text = snap.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = _RE_JANKY_FRAMES_PCT.search(text)
        if m:
            try:
                pcts.append(float(m.group(1)))
            except ValueError:
                pass
    if not pcts:
        return None
    return statistics.mean(pcts)


def parse_vmrss_delta_kb(run_dir: Path) -> Optional[int]:
    snaps = sorted(run_dir.glob("lmkd_status_*.txt"))
    if len(snaps) < 2:
        return None

    def vmrss(p: Path) -> Optional[int]:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        m = _RE_VMRSS.search(text)
        return int(m.group(1)) if m else None

    first = vmrss(snaps[0])
    last = vmrss(snaps[-1])
    if first is None or last is None:
        return None
    return last - first


def parse_coldstart_p50_ms(run_dir: Path) -> Optional[float]:
    """Median TotalTime across all probes in coldstart_pre/post files."""
    samples: list[int] = []
    for fname in ("coldstart_pre.txt", "coldstart_post.txt"):
        p = run_dir / fname
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _RE_TOTALTIME.finditer(text):
            try:
                samples.append(int(m.group(1)))
            except ValueError:
                pass
    if not samples:
        return None
    return float(statistics.median(samples))


def aggregate_run(run_dir: Path) -> Optional[dict]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    duration_sec = float(manifest.get("duration_sec") or 0.0)
    lmkd_log = run_dir / "lmkd.log"

    p50_ms, p99_ms = parse_inference_latency(lmkd_log)
    row = {
        "workload":         manifest.get("workload"),
        "ml":               manifest.get("ml"),
        "run":              manifest.get("run_index"),
        "jank_pct":         parse_jank_pct(run_dir),
        "kills_per_hour":   parse_kills_per_hour(lmkd_log, duration_sec),
        "inf_p50_ms":       p50_ms,
        "inf_p99_ms":       p99_ms,
        "vmrss_delta_kb":   parse_vmrss_delta_kb(run_dir),
        "coldstart_p50_ms": parse_coldstart_p50_ms(run_dir),
    }
    return row


# ---- entry ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate Phase 5 A/B raw outputs into a CSV.")
    p.add_argument(
        "--in", dest="in_glob", required=True,
        help="glob expanding to per-cell run directories (quote it)",
    )
    p.add_argument(
        "--out", dest="out_csv", required=True, type=Path,
        help="output summary CSV path",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    matches = [Path(p) for p in glob.glob(args.in_glob)]
    run_dirs = [p for p in matches if p.is_dir() and (p / "manifest.json").exists()]
    if not run_dirs:
        print(f"ERR: no run directories matched: {args.in_glob}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for rd in sorted(run_dirs):
        row = aggregate_run(rd)
        if row is None:
            print(f"WARN: skipped (no manifest): {rd}", file=sys.stderr)
            continue
        rows.append(row)

    df = pd.DataFrame(
        rows,
        columns=[
            "workload", "ml", "run",
            "jank_pct", "kills_per_hour",
            "inf_p50_ms", "inf_p99_ms",
            "vmrss_delta_kb", "coldstart_p50_ms",
        ],
    )
    df.sort_values(["workload", "ml", "run"], inplace=True, kind="stable")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {len(df)} rows -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
