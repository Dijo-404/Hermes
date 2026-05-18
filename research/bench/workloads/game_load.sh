#!/usr/bin/env bash
#
# game_load.sh — Phase 5 workload: high-memory app idle.
#
# Purpose
# -------
# Starts a high-memory ("game") app whose package name is supplied via
# the BENCH_GAME_PKG environment variable and lets it sit at the title
# screen for the run duration. The pre-allocated working set evicts
# everything else in cache and surfaces the LRU-tail kills the ML
# predictor is meant to anticipate.
#
# Usage
# -----
#   BENCH_GAME_PKG=com.example.game ./game_load.sh \
#       --device <serial> [--duration <seconds>]
#
# Exit codes
# ----------
#   0 workload ran to completion.
#   1 CLI / argument validation failure (incl. missing BENCH_GAME_PKG).
#   2 The game package failed to start (am start non-zero).
#
set -euo pipefail
IFS=$'\n\t'

DEVICE=""
DURATION=600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)   DEVICE="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        *) echo "[game_load] unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -n "${DEVICE}" ]] || { echo "[game_load] --device required" >&2; exit 1; }
[[ "${DURATION}" =~ ^[1-9][0-9]*$ ]] || { echo "[game_load] bad --duration" >&2; exit 1; }

if [[ -z "${BENCH_GAME_PKG:-}" ]]; then
    echo "[game_load] BENCH_GAME_PKG env var required (high-memory package)" >&2
    exit 1
fi

echo "[game_load] starting ${BENCH_GAME_PKG} on ${DEVICE}"
adb -s "${DEVICE}" shell "monkey -p ${BENCH_GAME_PKG} -c android.intent.category.LAUNCHER 1" \
    >/dev/null 2>&1 || exit 2

# Let the game allocate, then idle for the rest of the budget.
echo "[game_load] idling ${DURATION}s with ${BENCH_GAME_PKG} in foreground"
sleep "${DURATION}"
echo "[game_load] done"
