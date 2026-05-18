#!/usr/bin/env bash
#
# app_switch.sh — Phase 5 workload: round-robin app cycling.
#
# Purpose
# -------
# Cycles through 10 installed apps with `am start`, holds each in the
# foreground for a short dwell, then advances. Replays this loop for
# `--duration`. Stresses the cached-process LRU and is the canonical
# scenario where lmkd reclaim hits jank.
#
# Usage
# -----
#   ./app_switch.sh --device <serial> [--duration <seconds>]
#
# Optional env:
#   BENCH_SWITCH_PKGS comma-separated list of 10 package names. Default
#                     is a representative spread of system + AOSP apps.
#   BENCH_DWELL_MS    per-app foreground dwell in ms (default 3000).
#
# Exit codes
# ----------
#   0 workload ran to completion.
#   1 CLI / argument validation failure.
#
set -euo pipefail
IFS=$'\n\t'

DEVICE=""
DURATION=600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)   DEVICE="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        *) echo "[app_switch] unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -n "${DEVICE}" ]] || { echo "[app_switch] --device required" >&2; exit 1; }
[[ "${DURATION}" =~ ^[1-9][0-9]*$ ]] || { echo "[app_switch] bad --duration" >&2; exit 1; }

DEFAULT_PKGS="com.android.settings,com.android.chrome,com.android.calculator2,com.android.deskclock,com.android.contacts,com.android.camera2,com.android.gallery3d,com.android.music,com.android.calendar,com.android.email"
PKGS_CSV="${BENCH_SWITCH_PKGS:-${DEFAULT_PKGS}}"
DWELL_MS="${BENCH_DWELL_MS:-3000}"
DWELL_SEC="$(awk -v m="${DWELL_MS}" 'BEGIN{printf "%.3f", m/1000}')"

IFS=',' read -ra PKGS <<< "${PKGS_CSV}"
echo "[app_switch] cycling ${#PKGS[@]} apps on ${DEVICE} for ${DURATION}s"

end_ts=$(( $(date +%s) + DURATION ))
i=0
while (( $(date +%s) < end_ts )); do
    pkg="${PKGS[$(( i % ${#PKGS[@]} ))]}"
    adb -s "${DEVICE}" shell "monkey -p ${pkg} -c android.intent.category.LAUNCHER 1" \
        >/dev/null 2>&1 || true
    sleep "${DWELL_SEC}"
    i=$(( i + 1 ))
done

echo "[app_switch] done (${i} switches)"
