#!/usr/bin/env bash
#
# camera_burst.sh — Phase 5 workload: camera burst with music.
#
# Purpose
# -------
# Plays a long-running silent audio loop in the background (forces an
# audio service to stay resident, similar to a typical "music app +
# camera" user pattern) and repeatedly fires the camera shutter via
# KEYCODE_CAMERA. Camera processes have large RSS spikes; this is the
# stress case where pre-emptive reclaim should pay off.
#
# Usage
# -----
#   ./camera_burst.sh --device <serial> [--duration <seconds>]
#
# Optional env:
#   BENCH_CAMERA_PKG  camera package (default com.android.camera2).
#   BENCH_MUSIC_PKG   audio app package (default com.android.music).
#   BENCH_BURST_MS    delay between shutter presses, ms (default 600).
#
# Exit codes
# ----------
#   0 workload ran to completion.
#   1 CLI / argument validation failure.
#   2 Camera failed to start.
#
set -euo pipefail
IFS=$'\n\t'

DEVICE=""
DURATION=600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)   DEVICE="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        *) echo "[camera_burst] unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -n "${DEVICE}" ]] || { echo "[camera_burst] --device required" >&2; exit 1; }
[[ "${DURATION}" =~ ^[1-9][0-9]*$ ]] || { echo "[camera_burst] bad --duration" >&2; exit 1; }

CAMERA_PKG="${BENCH_CAMERA_PKG:-com.android.camera2}"
MUSIC_PKG="${BENCH_MUSIC_PKG:-com.android.music}"
BURST_MS="${BENCH_BURST_MS:-600}"
BURST_SEC="$(awk -v m="${BURST_MS}" 'BEGIN{printf "%.3f", m/1000}')"

echo "[camera_burst] starting ${MUSIC_PKG} background"
adb -s "${DEVICE}" shell "monkey -p ${MUSIC_PKG} -c android.intent.category.LAUNCHER 1" \
    >/dev/null 2>&1 || true
sleep 2

echo "[camera_burst] starting ${CAMERA_PKG}"
adb -s "${DEVICE}" shell "am start -a android.media.action.STILL_IMAGE_CAMERA" \
    >/dev/null 2>&1 || exit 2
sleep 3

end_ts=$(( $(date +%s) + DURATION ))
shots=0
while (( $(date +%s) < end_ts )); do
    adb -s "${DEVICE}" shell input keyevent KEYCODE_CAMERA >/dev/null 2>&1 || true
    sleep "${BURST_SEC}"
    shots=$(( shots + 1 ))
done

echo "[camera_burst] done (${shots} shutter events)"
