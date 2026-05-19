#!/usr/bin/env bash
#
# idle.sh — Phase 5 workload: idle baseline.
#
# Purpose
# -------
# Leaves the device on the home screen for `--duration` seconds. No
# foreground activity is triggered; this gives a noise floor for the
# A/B comparison and surfaces background-induced kills (e.g. ZRAM
# growth from system services) that aren't workload-correlated.
#
# Usage
# -----
#   ./idle.sh --device <serial> [--duration <seconds>]
#
# Defaults: --duration 600 (10 minutes per plan-executable.md §Phase 5).
#
# Exit codes
# ----------
#   0 ran for the full requested duration.
#   1 CLI / argument validation failure.
#   2 adb home-screen wake failed.
#
set -euo pipefail
IFS=$'\n\t'

DEVICE=""
DURATION=600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)   DEVICE="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        *) echo "[idle] unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -n "${DEVICE}" ]] || { echo "[idle] --device required" >&2; exit 1; }
[[ "${DURATION}" =~ ^[1-9][0-9]*$ ]] || { echo "[idle] bad --duration" >&2; exit 1; }

# Make sure screen is on and unlocked-as-much-as-we-can; on a research
# device the lockscreen is typically disabled.
adb -s "${DEVICE}" shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1 || exit 2
adb -s "${DEVICE}" shell input keyevent KEYCODE_HOME    >/dev/null 2>&1 || true

echo "[idle] idling for ${DURATION}s on ${DEVICE}"
sleep "${DURATION}"
echo "[idle] done"
