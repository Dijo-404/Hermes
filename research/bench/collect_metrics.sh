#!/usr/bin/env bash
#
# collect_metrics.sh — per-run telemetry collector for the Phase 5 harness.
#
# Purpose
# -------
# Spawned once per A/B cell by ab.sh. Owns metric collection for the
# duration of a single workload run and writes raw artifacts into the
# cell output directory. The aggregator (aggregate.py) parses these
# files; this script does NOT do any analysis.
#
# Files written into <out>:
#   - lmkd.log               : `logcat -s lmkd:I lmkd-ml:I` for the run.
#                              Contains both `Kill '<name>' ...` lines
#                              (lmkd.cpp:2506/2513) used as kill labels
#                              and `lmkd-ml: pre-emptive kill triggered`
#                              + `inference latency p50=... p99=...`
#                              lines from ml_predictor.cpp.
#   - lmkd_status_NNNN.txt   : snapshots of /proc/$(pidof lmkd)/status
#                              taken every 30s. Used for VmRSS delta.
#   - gfxinfo_<pkg>_NNNN.txt : `dumpsys gfxinfo <pkg>` snapshots every
#                              30s; parsed for janky-frames %.
#   - coldstart_pre.txt      : `am start -W` cold-start timings for 5
#                              apps measured BEFORE the workload starts.
#   - coldstart_post.txt     : same 5 apps measured AFTER the workload.
#   - meta.txt               : start/end timestamps and pid bookkeeping.
#
# Usage
# -----
#   ./collect_metrics.sh --device <serial> --out <dir> --duration <sec>
#
# Optional environment variables:
#   BENCH_GFX_PKG     comma-separated list of packages to sample with
#                     `dumpsys gfxinfo` (default: com.android.systemui,
#                     com.android.chrome).
#   BENCH_COLD_PKGS   comma-separated list of 5 packages used for cold-
#                     start probes (default: 5 widely-installed apps).
#   BENCH_SNAP_SEC    sampling cadence in seconds (default 30).
#
# Exit codes
# ----------
#   0 collection ran to completion (some samples may be missing if a
#     subprocess hiccuped; downstream parser is tolerant).
#   1 CLI / argument validation failure.
#   2 lmkd pid could not be resolved on the device.
#
set -euo pipefail
IFS=$'\n\t'

DEVICE=""
OUT=""
DURATION=""

SNAP_SEC="${BENCH_SNAP_SEC:-30}"
GFX_PKG_CSV="${BENCH_GFX_PKG:-com.android.systemui,com.android.chrome}"
COLD_PKG_CSV="${BENCH_COLD_PKGS:-com.android.settings,com.android.chrome,com.android.calculator2,com.android.deskclock,com.android.contacts}"

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

log() { printf '[collect %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
die() { local c="$1"; shift; printf '[collect ERROR] %s\n' "$*" >&2; exit "$c"; }

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device)   DEVICE="$2"; shift 2 ;;
            --out)      OUT="$2"; shift 2 ;;
            --duration) DURATION="$2"; shift 2 ;;
            -h|--help)  usage; exit 0 ;;
            *)          die 1 "unknown arg: $1" ;;
        esac
    done
    [[ -n "${DEVICE}"   ]] || die 1 "--device required"
    [[ -n "${OUT}"      ]] || die 1 "--out required"
    [[ -n "${DURATION}" ]] || die 1 "--duration required"
    [[ "${DURATION}" =~ ^[1-9][0-9]*$ ]] || die 1 "--duration must be positive integer"
    mkdir -p "${OUT}"
}

# Cold-start probe: am start -W returns TotalTime / WaitTime / etc. on
# stdout. We run each package once, force-stop in between to ensure a
# cold start, and append the raw `am start -W` block to <outfile>.
probe_coldstarts() {
    local outfile="$1"
    local pkg
    : > "${outfile}"
    IFS=',' read -ra COLD <<< "${COLD_PKG_CSV}"
    for pkg in "${COLD[@]}"; do
        adb -s "${DEVICE}" shell "am force-stop ${pkg}" >/dev/null 2>&1 || true
        sleep 1
        {
            echo "=== cold-start ${pkg} ==="
            adb -s "${DEVICE}" shell "am start -W ${pkg}" 2>&1 || true
            echo
        } >> "${outfile}"
        adb -s "${DEVICE}" shell "am force-stop ${pkg}" >/dev/null 2>&1 || true
    done
}

main() {
    parse_args "$@"
    : > "${OUT}/meta.txt"
    echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/meta.txt"
    echo "device=${DEVICE}"                          >> "${OUT}/meta.txt"
    echo "duration_sec=${DURATION}"                  >> "${OUT}/meta.txt"

    # 1) Pre-workload cold-start probe.
    log "pre-workload cold-start probe"
    probe_coldstarts "${OUT}/coldstart_pre.txt"

    # 2) Clear logcat so kill counts are scoped to this run.
    adb -s "${DEVICE}" logcat -c >/dev/null 2>&1 || true

    # 3) Start streaming lmkd + lmkd-ml logs.
    log "starting logcat tail"
    adb -s "${DEVICE}" logcat -s lmkd:I lmkd-ml:I \
        > "${OUT}/lmkd.log" 2>"${OUT}/lmkd.logcat.stderr" &
    local logcat_pid=$!

    # 4) Resolve lmkd pid for /proc/<pid>/status snapshots.
    local lmkd_pid
    lmkd_pid="$(adb -s "${DEVICE}" shell 'pidof lmkd' 2>/dev/null | tr -d '\r\n' || true)"
    if [[ -z "${lmkd_pid}" ]]; then
        kill "${logcat_pid}" >/dev/null 2>&1 || true
        die 2 "could not resolve lmkd pid"
    fi
    echo "lmkd_pid=${lmkd_pid}" >> "${OUT}/meta.txt"

    # 5) Sampling loop. Every SNAP_SEC seconds, capture status + gfxinfo.
    IFS=',' read -ra GFX <<< "${GFX_PKG_CSV}"
    local end_ts=$(( $(date +%s) + DURATION ))
    local idx=0
    while (( $(date +%s) < end_ts )); do
        local stamp
        stamp="$(printf '%04d' "${idx}")"
        # /proc/<pid>/status — first 30 lines is sufficient for VmRSS, VmSize.
        adb -s "${DEVICE}" shell "cat /proc/${lmkd_pid}/status 2>/dev/null" \
            > "${OUT}/lmkd_status_${stamp}.txt" 2>/dev/null || true

        local pkg
        for pkg in "${GFX[@]}"; do
            adb -s "${DEVICE}" shell "dumpsys gfxinfo ${pkg}" \
                > "${OUT}/gfxinfo_${pkg}_${stamp}.txt" 2>/dev/null || true
        done

        idx=$(( idx + 1 ))
        sleep "${SNAP_SEC}"
    done

    # 6) Stop log tail.
    kill "${logcat_pid}" >/dev/null 2>&1 || true
    wait "${logcat_pid}" 2>/dev/null || true

    # 7) Post-workload cold-start probe.
    log "post-workload cold-start probe"
    probe_coldstarts "${OUT}/coldstart_post.txt"

    echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/meta.txt"
    echo "samples=${idx}"                          >> "${OUT}/meta.txt"
    log "collection complete (${idx} snapshots)"
}

main "$@"
