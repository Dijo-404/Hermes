#!/usr/bin/env python3
"""
analyze.py — Phase 5 statistical analysis & gate report.

Purpose
-------
Consumes the summary CSV produced by aggregate.py and emits a Markdown
report comparing the `ml=on` arm against the `ml=off` arm per workload.
For each workload:

  - Bootstrap 95% CI (N=1000 resamples) on the jank-percent delta
    (ml_on - ml_off) and on the kills_per_hour delta.
  - Evaluate the four success gates from plan-executable.md Phase 5 /
    plan.md §2, sourced as constants at the top of this file so they
    are visible to reviewers and easy to retune.
  - Aggregate verdict across all workloads (PASS iff every workload
    passes every gate).

Success gate constants (sourced from plan-executable.md §Phase 5,
plan.md §2 success criteria):

    JANK_DELTA_PCT_MAX   = -30.0  # ml_on jank must be >=30% lower than ml_off
    KILLS_DELTA_PCT_MAX  =  +5.0  # kill rate may rise at most 5%
    INF_P99_MS_MAX       =   2.0  # plan.md inference latency budget
    VMRSS_DELTA_KB_MAX   =  4096  # 4 MiB upper bound on lmkd memory growth

A "delta_pct" is computed as (mean(ml_on) - mean(ml_off)) / mean(ml_off)
* 100. A negative value means the ML arm did better (fewer jank frames,
fewer kills, smaller RSS).

Usage
-----
    python analyze.py --summary <summary-csv> --out <report-md>

Exit codes
----------
  0 report written; aggregate verdict was PASS.
  1 CLI / argument validation failure.
  2 summary CSV missing / empty.
  3 report written; aggregate verdict was FAIL (still produces output).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# --- success gates (plan-executable.md §Phase 5, plan.md §2) --------------

JANK_DELTA_PCT_MAX:  float = -30.0   # >=30% reduction required
KILLS_DELTA_PCT_MAX: float = +5.0    # <=5% increase permitted
INF_P99_MS_MAX:      float = 2.0     # ms
VMRSS_DELTA_KB_MAX:  int   = 4096    # kB

BOOTSTRAP_N:   int = 1000
BOOTSTRAP_CI:  float = 0.95
BOOTSTRAP_SEED: int = 20260518   # fix for reproducibility across reruns


# --- statistics -----------------------------------------------------------

@dataclass
class CI:
    point:  float
    low:    float
    high:   float

    def fmt(self, unit: str = "") -> str:
        return f"{self.point:+.3f}{unit} (95% CI {self.low:+.3f}{unit} .. {self.high:+.3f}{unit})"


def bootstrap_delta_ci(
    on: np.ndarray,
    off: np.ndarray,
    n: int = BOOTSTRAP_N,
    ci: float = BOOTSTRAP_CI,
    seed: int = BOOTSTRAP_SEED,
) -> Optional[CI]:
    """Bootstrap CI on mean(on) - mean(off). Returns None if any input empty."""
    on = on[~np.isnan(on)]
    off = off[~np.isnan(off)]
    if on.size == 0 or off.size == 0:
        return None
    rng = np.random.default_rng(seed)
    point = float(on.mean() - off.mean())
    deltas = np.empty(n, dtype=np.float64)
    for i in range(n):
        s_on  = rng.choice(on,  size=on.size,  replace=True)
        s_off = rng.choice(off, size=off.size, replace=True)
        deltas[i] = s_on.mean() - s_off.mean()
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    return CI(
        point=point,
        low=float(np.quantile(deltas, lo_q)),
        high=float(np.quantile(deltas, hi_q)),
    )


def pct_delta(on_mean: float, off_mean: float) -> Optional[float]:
    if off_mean == 0 or np.isnan(off_mean) or np.isnan(on_mean):
        return None
    return (on_mean - off_mean) / off_mean * 100.0


# --- gate evaluation ------------------------------------------------------

@dataclass
class GateResult:
    name:    str
    value:   Optional[float]
    bound:   float
    ok:      bool
    note:    str = ""


def evaluate_gates(
    jank_pct_delta:  Optional[float],
    kills_pct_delta: Optional[float],
    inf_p99_ms_on:   Optional[float],
    vmrss_delta_kb:  Optional[float],
) -> list[GateResult]:
    gates: list[GateResult] = []

    # jank: must be <= -30%
    gates.append(GateResult(
        name="jank_delta_pct <= -30%",
        value=jank_pct_delta,
        bound=JANK_DELTA_PCT_MAX,
        ok=(jank_pct_delta is not None and jank_pct_delta <= JANK_DELTA_PCT_MAX),
    ))
    # kills: must be <= +5%
    gates.append(GateResult(
        name="kills_delta_pct <= +5%",
        value=kills_pct_delta,
        bound=KILLS_DELTA_PCT_MAX,
        ok=(kills_pct_delta is not None and kills_pct_delta <= KILLS_DELTA_PCT_MAX),
    ))
    # inference p99 (ML arm only): <= 2 ms
    gates.append(GateResult(
        name="inf_p99_ms <= 2.0",
        value=inf_p99_ms_on,
        bound=INF_P99_MS_MAX,
        ok=(inf_p99_ms_on is not None and inf_p99_ms_on <= INF_P99_MS_MAX),
    ))
    # VmRSS delta (ML arm): <= 4096 kB
    gates.append(GateResult(
        name="vmrss_delta_kb <= 4096",
        value=vmrss_delta_kb,
        bound=float(VMRSS_DELTA_KB_MAX),
        ok=(vmrss_delta_kb is not None and vmrss_delta_kb <= VMRSS_DELTA_KB_MAX),
    ))
    return gates


# --- report rendering -----------------------------------------------------

def _fmt(v: Optional[float], digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def render_workload_section(
    workload: str,
    on_df:  pd.DataFrame,
    off_df: pd.DataFrame,
) -> tuple[list[str], list[GateResult]]:
    lines: list[str] = []
    lines.append(f"### Workload: `{workload}`")
    lines.append("")
    lines.append(f"- runs (ml_on)  : {len(on_df)}")
    lines.append(f"- runs (ml_off) : {len(off_df)}")
    lines.append("")

    # Means table.
    def col_means(df: pd.DataFrame, c: str) -> float:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        return float(s.mean()) if len(s) else float("nan")

    metrics = ["jank_pct", "kills_per_hour", "inf_p50_ms", "inf_p99_ms",
               "vmrss_delta_kb", "coldstart_p50_ms"]
    lines.append("| metric | ml_off mean | ml_on mean | delta % |")
    lines.append("|---|---:|---:|---:|")
    on_means  = {m: col_means(on_df,  m) for m in metrics}
    off_means = {m: col_means(off_df, m) for m in metrics}
    for m in metrics:
        d = pct_delta(on_means[m], off_means[m])
        lines.append(
            f"| {m} | {_fmt(off_means[m])} | {_fmt(on_means[m])} | "
            f"{_fmt(d)} |"
        )
    lines.append("")

    # Bootstrap CIs on the two headline deltas.
    jank_ci = bootstrap_delta_ci(
        pd.to_numeric(on_df["jank_pct"],  errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(off_df["jank_pct"], errors="coerce").to_numpy(dtype=np.float64),
    )
    kills_ci = bootstrap_delta_ci(
        pd.to_numeric(on_df["kills_per_hour"],  errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(off_df["kills_per_hour"], errors="coerce").to_numpy(dtype=np.float64),
    )
    lines.append("**Bootstrap 95% CI on ml_on - ml_off (N=1000 resamples)**")
    lines.append("")
    lines.append(f"- jank_pct delta       : {jank_ci.fmt(' pp') if jank_ci else 'n/a'}")
    lines.append(f"- kills_per_hour delta : {kills_ci.fmt() if kills_ci else 'n/a'}")
    lines.append("")

    # Gate evaluation.
    gates = evaluate_gates(
        jank_pct_delta=pct_delta(on_means["jank_pct"],       off_means["jank_pct"]),
        kills_pct_delta=pct_delta(on_means["kills_per_hour"], off_means["kills_per_hour"]),
        inf_p99_ms_on=on_means.get("inf_p99_ms"),
        vmrss_delta_kb=on_means.get("vmrss_delta_kb"),
    )
    lines.append("**Success gates**")
    lines.append("")
    lines.append("| gate | observed | bound | result |")
    lines.append("|---|---:|---:|:---:|")
    for g in gates:
        lines.append(
            f"| {g.name} | {_fmt(g.value)} | {_fmt(g.bound)} | "
            f"{'PASS' if g.ok else 'FAIL'} |"
        )
    lines.append("")
    return lines, gates


def render_report(df: pd.DataFrame) -> tuple[str, bool]:
    out: list[str] = []
    out.append("# Phase 5 A/B Benchmark Report")
    out.append("")
    out.append(f"- summary rows           : {len(df)}")
    out.append(f"- workloads              : {sorted(df['workload'].unique().tolist())}")
    out.append(f"- bootstrap resamples    : {BOOTSTRAP_N}")
    out.append(f"- bootstrap CI level     : {int(BOOTSTRAP_CI*100)}%")
    out.append("")
    out.append("Success gates (constants — see top of analyze.py):")
    out.append("")
    out.append(f"- `JANK_DELTA_PCT_MAX  = {JANK_DELTA_PCT_MAX}`  (>=30% jank reduction)")
    out.append(f"- `KILLS_DELTA_PCT_MAX = {KILLS_DELTA_PCT_MAX}` (<=5% kill increase)")
    out.append(f"- `INF_P99_MS_MAX      = {INF_P99_MS_MAX}`  (inference latency budget)")
    out.append(f"- `VMRSS_DELTA_KB_MAX  = {VMRSS_DELTA_KB_MAX}` (lmkd RSS growth ceiling)")
    out.append("")

    all_pass = True
    for workload in sorted(df["workload"].dropna().unique()):
        sub = df[df["workload"] == workload]
        on_df  = sub[sub["ml"] == "on"]
        off_df = sub[sub["ml"] == "off"]
        section, gates = render_workload_section(workload, on_df, off_df)
        out.extend(section)
        if not all(g.ok for g in gates):
            all_pass = False

    out.append("---")
    out.append("")
    out.append(f"## Aggregate verdict: **{'PASS' if all_pass else 'FAIL'}**")
    out.append("")
    out.append("A run is PASS only when every workload clears every gate above.")
    return "\n".join(out) + "\n", all_pass


# --- entry ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Phase 5 summary CSV.")
    p.add_argument("--summary", required=True, type=Path)
    p.add_argument("--out",     required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.summary.exists():
        print(f"ERR: summary CSV missing: {args.summary}", file=sys.stderr)
        return 2
    df = pd.read_csv(args.summary)
    if df.empty:
        print(f"ERR: summary CSV has no rows: {args.summary}", file=sys.stderr)
        return 2

    report, all_pass = render_report(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote report -> {args.out}  verdict={'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 3


if __name__ == "__main__":
    sys.exit(main())
