#!/usr/bin/env bash
#
# web_scroll.sh — Phase 5 workload: Chrome 30-tab scroll.
#
# Purpose
# -------
# Opens 30 Chrome tabs (one per URL in the rotation), waits for each
# tab to commit, then programmatically scrolls the active tab using
# `input swipe` events for the rest of the duration. Heavy on
# foreground-process working set and triggers the kinds of background
# kills (cached app reclaim) that the ML predictor is hypothesised to
# catch ahead of jank.
#
# Usage
# -----
#   ./web_scroll.sh --device <serial> [--duration <seconds>]
#
# Optional env:
#   BENCH_WEB_URLS  comma-separated URL list. Default: 30 lightweight
#                   public pages tiled to fill the rotation.
#
# Exit codes
# ----------
#   0 workload ran to completion.
#   1 CLI / argument validation failure.
#   2 Chrome failed to start (am start returned non-zero).
#
set -euo pipefail
IFS=$'\n\t'

DEVICE=""
DURATION=600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)   DEVICE="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        *) echo "[web_scroll] unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -n "${DEVICE}" ]] || { echo "[web_scroll] --device required" >&2; exit 1; }
[[ "${DURATION}" =~ ^[1-9][0-9]*$ ]] || { echo "[web_scroll] bad --duration" >&2; exit 1; }

DEFAULT_URLS="https://en.wikipedia.org/wiki/Linux,https://en.wikipedia.org/wiki/Android_(operating_system),https://en.wikipedia.org/wiki/Memory_management,https://en.wikipedia.org/wiki/Page_replacement_algorithm,https://en.wikipedia.org/wiki/Out-of-memory,https://en.wikipedia.org/wiki/Cgroups,https://en.wikipedia.org/wiki/Epoll,https://en.wikipedia.org/wiki/Pixel_(smartphone),https://en.wikipedia.org/wiki/Snapdragon,https://en.wikipedia.org/wiki/ARM_architecture,https://en.wikipedia.org/wiki/ONNX,https://en.wikipedia.org/wiki/Long_short-term_memory,https://en.wikipedia.org/wiki/Reproducibility,https://en.wikipedia.org/wiki/Pressure_stall_information,https://en.wikipedia.org/wiki/Zram,https://en.wikipedia.org/wiki/Swap_(computing),https://en.wikipedia.org/wiki/Kernel_(operating_system),https://en.wikipedia.org/wiki/Process_(computing),https://en.wikipedia.org/wiki/Linux_kernel,https://en.wikipedia.org/wiki/SystemD,https://en.wikipedia.org/wiki/Init,https://en.wikipedia.org/wiki/Binder_(IPC),https://en.wikipedia.org/wiki/Dalvik_(software),https://en.wikipedia.org/wiki/ART_(Android_Runtime),https://en.wikipedia.org/wiki/Vulkan_(API),https://en.wikipedia.org/wiki/GPU,https://en.wikipedia.org/wiki/Refresh_rate,https://en.wikipedia.org/wiki/Frame_rate,https://en.wikipedia.org/wiki/Janky,https://en.wikipedia.org/wiki/Web_browser"
URLS_CSV="${BENCH_WEB_URLS:-${DEFAULT_URLS}}"

IFS=',' read -ra URLS <<< "${URLS_CSV}"

echo "[web_scroll] opening ${#URLS[@]} Chrome tabs on ${DEVICE}"
# First URL launches Chrome; subsequent VIEWs open new tabs.
adb -s "${DEVICE}" shell am start -W -a android.intent.action.VIEW -d "${URLS[0]}" \
    com.android.chrome >/dev/null 2>&1 || exit 2
sleep 4
for ((i=1; i<${#URLS[@]}; i++)); do
    adb -s "${DEVICE}" shell am start -a android.intent.action.VIEW \
        -d "${URLS[$i]}" com.android.chrome >/dev/null 2>&1 || true
    sleep 1
done

# Compute scroll loop budget. Each scroll cycle is ~0.5s (swipe duration
# 300ms + sleep 200ms). End when we hit DURATION.
end_ts=$(( $(date +%s) + DURATION ))
echo "[web_scroll] scrolling for $(( end_ts - $(date +%s) ))s"
while (( $(date +%s) < end_ts )); do
    adb -s "${DEVICE}" shell input swipe 500 1500 500 500 300 >/dev/null 2>&1 || true
    sleep 0.2
    adb -s "${DEVICE}" shell input swipe 500 500 500 1500 300 >/dev/null 2>&1 || true
    sleep 0.2
done

echo "[web_scroll] done"
