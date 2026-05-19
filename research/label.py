#!/usr/bin/env python3
"""
label.py — post-process collector CSV with kill-event labels.

Purpose
-------
For each kill instant T_k recorded by collector.py, sets kill_event=1 on
every row whose timestamp_unix falls in the predict-ahead window
[T_k - lead_ms - window_ms, T_k - lead_ms]. Defaults: lead=200ms,
window=100ms, matching plan-executable.md Phase 2 task 3.

Usage
-----
  python label.py --in data/web.csv \\
                  --kills data/web.csv.kills.log \\
                  --out  data/web.labeled.csv

Exit codes
----------
  0 — labeled CSV written.
  1 — missing input file.
  2 — input CSV missing required columns.
"""

from __future__ import annotations

import argparse
import csv
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

REQUIRED_COLUMNS: set[str] = {"timestamp_unix", "kill_event"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply kill_event labels to a collector CSV.")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--kills", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--lead-ms", type=int, default=200)
    p.add_argument("--window-ms", type=int, default=100)
    return p.parse_args()


def load_kills(path: Path) -> list[float]:
    """One epoch-seconds float per line. Blank lines / comments tolerated."""
    out: list[float] = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        try:
            out.append(float(s))
        except ValueError:
            sys.stderr.write(f"warn: skipping unparsable kill timestamp {s!r}\n")
    out.sort()
    return out


def label_rows(
    rows: list[dict[str, str]],
    kills: list[float],
    lead_s: float,
    window_s: float,
) -> int:
    """Mutate rows in-place. Returns the count of positive labels."""
    timestamps = [float(r["timestamp_unix"]) for r in rows]
    positives = 0
    for k in kills:
        lo = k - lead_s - window_s
        hi = k - lead_s
        i = bisect_left(timestamps, lo)
        j = bisect_right(timestamps, hi)
        for idx in range(i, j):
            if rows[idx]["kill_event"] != "1":
                rows[idx]["kill_event"] = "1"
                positives += 1
    return positives


def main() -> int:
    args = parse_args()
    if not args.inp.is_file():
        sys.stderr.write(f"error: input CSV not found: {args.inp}\n")
        return 1
    if not args.kills.is_file():
        sys.stderr.write(f"error: kill log not found: {args.kills}\n")
        return 1

    with args.inp.open(newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            sys.stderr.write(f"error: CSV missing columns: {sorted(missing)}\n")
            return 2
        rows = list(reader)

    kills = load_kills(args.kills)
    lead_s = args.lead_ms / 1000.0
    window_s = args.window_ms / 1000.0
    positives = label_rows(rows, kills, lead_s, window_s)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    frac = positives / total if total else 0.0
    sys.stderr.write(
        f"label: rows={total} kills={len(kills)} positives={positives} "
        f"fraction={frac:.4f}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
