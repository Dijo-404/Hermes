#!/usr/bin/env python3
"""
collector.py — PSI / meminfo sampling pipeline for ML-driven lmkd training data.

Purpose
-------
Samples /proc/pressure/memory and /proc/meminfo on an ADB-connected Android
device every ~100 ms, while concurrently streaming `logcat -s lmkd:I` to
capture kill events (statslog.cpp:298 lmkd_pack_set_kill_occurred). Emits
two artifacts:

  1. A CSV of sampled rows (one per 100 ms tick) at the path given by --out.
  2. A side-channel kill-log of one Unix-epoch timestamp per line at
     <out>.kills.log. Feed both to `label.py` to populate kill_event labels.

This collector intentionally does NOT label rows in real time — labels need
a future-window lookup (T_k - 200ms .. T_k - 100ms) which is cleaner as a
post-pass.

Design note: per-sample `adb shell` invocations cost ~30–60 ms of round-trip
overhead each, which would skew a 100 ms cadence (see plan-executable.md
Phase 2 anti-pattern). We therefore spawn ONE persistent `adb shell` and run
a tiny on-device loop that prints a sentinel-delimited record per tick; the
host parses stdout. If the persistent shell wedges, run with --no-persist to
fall back to per-sample `adb exec-out cat` (correctness preserved, cadence
degrades to ~150–200 ms).

Usage
-----
  python collector.py --out data/idle.csv --duration 300 --scenario idle
  python collector.py --out data/web.csv  --duration 600 --device 0A111JECB
                      --scenario web_scroll

Exit codes
----------
  0 — clean shutdown (duration elapsed or Ctrl-C with CSV flushed).
  1 — adb not on PATH, or device not found.
  2 — persistent shell failed to produce any samples within 5 s.
  3 — CSV write error.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TextIO

# --- CSV schema (kept in lockstep with label.py and eda.ipynb) -----------------
CSV_COLUMNS: list[str] = [
    "timestamp_unix",
    "scenario",
    "some_avg10",
    "some_avg60",
    "some_avg300",
    "some_total",
    "full_avg10",
    "full_avg60",
    "full_avg300",
    "full_total",
    "mem_available_kb",
    "swap_free_kb",
    "swap_total_kb",
    "kill_event",
]

# Matches the libpsi/psi.cpp:109 parse format:
#   some avg10=X.XX avg60=X.XX avg300=X.XX total=NNNNN
PSI_RE = re.compile(
    r"^(some|full)\s+avg10=([\d.]+)\s+avg60=([\d.]+)\s+avg300=([\d.]+)\s+total=(\d+)"
)

# Sentinel framing for the persistent on-device loop.
REC_BEGIN = "###REC_BEGIN###"
REC_END = "###REC_END###"

# Tolerant kill-line regex. `logcat -v epoch` prefixes each line with
# `<seconds>.<micros>  <pid> <tid> <level> <tag>: <msg>`. We capture the
# epoch and require "Kill" or "killed" anywhere in the message body — the
# exact wording from lmkd_pack_set_kill_occurred varies across Android
# releases ("Kill ...", "Killing ...", "killed pid=..."), so we stay loose
# on the verb but anchor on the lmkd tag (already filtered by `-s lmkd:I`).
KILL_RE = re.compile(
    r"^\s*(\d+\.\d+)\s+\d+\s+\d+\s+[VDIWEF]\s+lmkd\s*:\s*.*\b(?:[Kk]ill(?:ed|ing)?)\b"
)


@dataclass
class CollectorState:
    """Mutable state shared between the sampler and logcat threads."""

    csv_writer: csv.writer
    csv_file: TextIO
    kill_log: TextIO
    scenario: str
    stop: threading.Event = field(default_factory=threading.Event)
    rows_written: int = 0
    kills_seen: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


# --- On-device shell loop ------------------------------------------------------
# Prints a framed record every ~100ms with the device's wall-clock epoch and
# the raw contents of /proc/pressure/memory and the three meminfo lines we
# care about. Kept short — single quotes on the host side, double on device.
ON_DEVICE_LOOP = (
    "while true; do "
    f"  echo {REC_BEGIN}; "
    "  date +%s.%N; "
    "  cat /proc/pressure/memory; "
    "  grep -E '^(MemAvailable|SwapFree|SwapTotal):' /proc/meminfo; "
    f"  echo {REC_END}; "
    "  sleep 0.1; "
    "done"
)


def adb_argv(device: Optional[str], *rest: str) -> list[str]:
    """Build an adb command line, optionally pinned to a device serial."""
    argv = ["adb"]
    if device:
        argv += ["-s", device]
    argv += list(rest)
    return argv


def spawn_sampler(device: Optional[str]) -> subprocess.Popen[str]:
    """Spawn the persistent adb shell that prints framed records."""
    return subprocess.Popen(
        adb_argv(device, "shell", ON_DEVICE_LOOP),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def spawn_logcat(device: Optional[str]) -> subprocess.Popen[str]:
    """Stream lmkd kill events with epoch timestamps."""
    # `-T 1` starts from "now" so we don't replay old kills.
    return subprocess.Popen(
        adb_argv(device, "logcat", "-b", "main", "-s", "lmkd:I", "-v", "epoch", "-T", "1"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def parse_record(block: list[str]) -> Optional[dict[str, float | int | str]]:
    """Parse a single framed record into a CSV-ready row dict."""
    if len(block) < 4:
        return None
    try:
        ts = float(block[0].strip())
    except ValueError:
        return None

    row: dict[str, float | int | str] = {c: "" for c in CSV_COLUMNS}
    row["timestamp_unix"] = ts
    row["kill_event"] = 0

    for line in block[1:]:
        m = PSI_RE.match(line.strip())
        if m:
            prefix = m.group(1)  # "some" or "full"
            row[f"{prefix}_avg10"] = float(m.group(2))
            row[f"{prefix}_avg60"] = float(m.group(3))
            row[f"{prefix}_avg300"] = float(m.group(4))
            row[f"{prefix}_total"] = int(m.group(5))
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            val_kb = val.strip().split()[0] if val.strip() else ""
            if key == "MemAvailable":
                row["mem_available_kb"] = val_kb
            elif key == "SwapFree":
                row["swap_free_kb"] = val_kb
            elif key == "SwapTotal":
                row["swap_total_kb"] = val_kb
    return row


def sampler_thread(proc: subprocess.Popen[str], state: CollectorState) -> None:
    """Read framed records from the persistent shell and write CSV rows."""
    assert proc.stdout is not None
    block: list[str] = []
    in_record = False
    for line in proc.stdout:
        if state.stop.is_set():
            break
        line = line.rstrip("\r\n")
        if line == REC_BEGIN:
            block = []
            in_record = True
            continue
        if line == REC_END and in_record:
            in_record = False
            row = parse_record(block)
            if row is not None:
                row["scenario"] = state.scenario
                with state.lock:
                    state.csv_writer.writerow([row[c] for c in CSV_COLUMNS])
                    state.rows_written += 1
            continue
        if in_record:
            block.append(line)


def logcat_thread(proc: subprocess.Popen[str], state: CollectorState) -> None:
    """Capture lmkd kill timestamps to the side-channel log."""
    assert proc.stdout is not None
    for line in proc.stdout:
        if state.stop.is_set():
            break
        m = KILL_RE.match(line)
        if m:
            with state.lock:
                state.kill_log.write(f"{m.group(1)}\n")
                state.kill_log.flush()
                state.kills_seen += 1


def preflight(device: Optional[str]) -> None:
    """Verify adb is on PATH and the target device is reachable."""
    if shutil.which("adb") is None:
        sys.stderr.write("error: 'adb' not found on PATH\n")
        sys.exit(1)
    out = subprocess.run(
        adb_argv(device, "get-state"), capture_output=True, text=True
    )
    if out.returncode != 0 or "device" not in out.stdout:
        sys.stderr.write(f"error: device not ready: {out.stdout!r} {out.stderr!r}\n")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ADB-driven PSI/meminfo collector.")
    p.add_argument("--out", required=True, type=Path, help="CSV output path.")
    p.add_argument("--duration", required=True, type=int, help="Run length (s).")
    p.add_argument("--device", default=None, help="ADB device serial.")
    p.add_argument("--scenario", default="unknown", help="Workload label.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    preflight(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kill_log_path = args.out.with_suffix(args.out.suffix + ".kills.log")

    sampler = spawn_sampler(args.device)
    logcat = spawn_logcat(args.device)

    try:
        with args.out.open("w", newline="") as csv_fp, kill_log_path.open("w") as kill_fp:
            writer = csv.writer(csv_fp)
            writer.writerow(CSV_COLUMNS)
            state = CollectorState(writer, csv_fp, kill_fp, args.scenario)

            t_samp = threading.Thread(target=sampler_thread, args=(sampler, state), daemon=True)
            t_log = threading.Thread(target=logcat_thread, args=(logcat, state), daemon=True)
            t_samp.start()
            t_log.start()

            def _handle_sigint(_signum: int, _frame: object) -> None:
                state.stop.set()

            signal.signal(signal.SIGINT, _handle_sigint)

            deadline = time.monotonic() + args.duration
            startup = time.monotonic() + 5.0
            while time.monotonic() < deadline and not state.stop.is_set():
                time.sleep(0.5)
                if time.monotonic() > startup and state.rows_written == 0:
                    sys.stderr.write("error: no samples after 5s — shell wedged\n")
                    state.stop.set()
                    return 2

            state.stop.set()
            csv_fp.flush()
            kill_fp.flush()
            sys.stderr.write(
                f"collector: rows={state.rows_written} kills={state.kills_seen}\n"
            )
    except OSError as e:
        sys.stderr.write(f"error: CSV write failed: {e}\n")
        return 3
    finally:
        for proc in (sampler, logcat):
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
