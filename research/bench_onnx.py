#!/usr/bin/env python3
"""
bench_onnx.py — micro-benchmark single-sample ONNX inference latency.

Purpose
-------
Runs 10,000 single-sample (batch=1, 20×6) inferences through an ONNX
model loaded by `onnxruntime.InferenceSession` and reports p50 / p99
latency in milliseconds. Phase 4's verification gate is p99 ≤ 2 ms on
Pixel-4a-class ARM64 hardware (plan-executable.md Phase 3 verification +
Phase 4 inference-budget constraint).

On a workstation, the absolute numbers are not meaningful but the script
verifies the inference graph runs at all and gives a relative comparison
point when porting to the ARM64 dev board.

Usage
-----
    python bench_onnx.py --onnx out/psi_predictor.onnx [--iters 10000] \\
                         [--warmup 200] [--seed 0]

Exit codes
----------
  0 — benchmark ran; latency summary printed.
  1 — ONNX file missing / unreadable.
  2 — onnxruntime InferenceSession failed to initialise.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


INPUT_NAME: str = "psi_window"
OUTPUT_NAME: str = "kill_prob"
WINDOW_LEN: int = 20
FEATURES: int = 6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark single-sample ONNX inference latency."
    )
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--iters", type=int, default=10_000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.onnx.exists():
        print(f"ERR: ONNX file not found: {args.onnx}", file=sys.stderr)
        return 1

    try:
        import onnxruntime as ort  # type: ignore
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(
            args.onnx.as_posix(),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERR: onnxruntime init failed: {e}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    inputs = [
        rng.standard_normal((1, WINDOW_LEN, FEATURES)).astype(np.float32)
        for _ in range(args.iters)
    ]

    # Warmup: amortise allocator + thread-pool startup.
    for i in range(args.warmup):
        sess.run([OUTPUT_NAME], {INPUT_NAME: inputs[i % len(inputs)]})

    times_ns = np.empty(args.iters, dtype=np.int64)
    for i in range(args.iters):
        t0 = time.perf_counter_ns()
        sess.run([OUTPUT_NAME], {INPUT_NAME: inputs[i]})
        times_ns[i] = time.perf_counter_ns() - t0

    times_ms = times_ns / 1_000_000.0
    p50 = float(np.percentile(times_ms, 50))
    p90 = float(np.percentile(times_ms, 90))
    p99 = float(np.percentile(times_ms, 99))
    p999 = float(np.percentile(times_ms, 99.9))
    mn = float(times_ms.min())
    mx = float(times_ms.max())
    print(
        f"iters={args.iters}  min={mn:.3f}ms  p50={p50:.3f}ms  "
        f"p90={p90:.3f}ms  p99={p99:.3f}ms  p99.9={p999:.3f}ms  "
        f"max={mx:.3f}ms"
    )
    print(f"Budget: p99 ≤ 2 ms (target hardware: Pixel 4a / ARM64).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
