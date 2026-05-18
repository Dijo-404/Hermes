#!/usr/bin/env bash
#
# ab.sh — Phase 5 A/B benchmark harness for the lmkd ML PSI predictor.
#
# Purpose
# -------
# Drives a paired A/B comparison of an lmkd binary built with
# LMKD_USE_ML=true against itself with the runtime property
# `ro.lmk.use_ml_predictor` flipped between `true` (ml_on) and `false`
# (ml_off). The same binary is used in both arms — per plan-executable.md
# Phase 5 anti-pattern row 1, the baseline must NOT be a different build.
#
# Each cell of the matrix (workload x runs x {ml_on, ml_off}) is run in a
# freshly rebooted device session. PSI state carries across workloads;
# Phase 5 anti-pattern row 2 forbids running baseline and experimental
# back-to-back in the same boot.
#
# Usage
# -----
#   ./ab.sh --device <serial> \
#           --workloads idle,web_scroll,app_switch,camera_burst,game_load \
#           --runs 5 \
#           --out ./results \
#           --build-id <build-fingerprint>
#
# Required:
#   --device      adb serial of the target device.
#   --workloads   comma-separated list of workload script names (without
#                 .sh) under research/bench/workloads/.
#   --runs        integer >= 1, number of repetitions per cell.
#   --out         output directory; one subdir per run-id will be created.
#   --build-id    free-form string written into each run's manifest for
#                 traceability (e.g. `git rev-parse HEAD`).
#
# Optional environment variables:
#   BENCH_DURATION_SEC   per-workload duration in seconds (default 600).
#   BENCH_REBOOT_TIMEOUT seconds to wait for sys.boot_completed (default 90).
#   BENCH_ML_THRESHOLD   forwarded to ro.lmk.ml_threshold (default 0.65).
#   BENCH_GAME_PKG       package name for the game_load workload.
#
# Exit codes
# ----------
#   0  All cells completed and metrics collected.
#   1  CLI / argument validation failure.
#   2  adb device not found or unauthorized.
#   3  Reboot wait timed out (sys.boot_completed never reached 1).
#   4  Workload script missing or non-executable.
#   5  Metrics collection failed for at least one cell (partial results
#      still written to <out>/).
#
set -euo pipefail
IFS=$'\n\t'

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKLOAD_DIR="${SCRIPT_DIR}/workloads"
readonly COLLECT_SCRIPT="${SCRIPT_DIR}/collect_metrics.sh"

DEVICE=""
WORKLOADS_CSV=""
RUNS=""
OUT_DIR=""
BUILD_ID=""

DURATION_SEC="${BENCH_DURATION_SEC:-600}"
REBOOT_TIMEOUT="${BENCH_REBOOT_TIMEOUT:-90}"
ML_THRESHOLD="${BENCH_ML_THRESHOLD:-0.65}"

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

log() {
    printf '[ab %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2
}

die() {
    local code="$1"; shift
    printf '[ab ERROR] %s\n' "$*" >&2
    exit "${code}"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device)     DEVICE="$2"; shift 2 ;;
            --workloads)  WORKLOADS_CSV="$2"; shift 2 ;;
            --runs)       RUNS="$2"; shift 2 ;;
            --out)        OUT_DIR="$2"; shift 2 ;;
            --build-id)   BUILD_ID="$2"; shift 2 ;;
            -h|--help)    usage; exit 0 ;;
            *)            die 1 "unknown arg: $1" ;;
        esac
    done

    [[ -n "${DEVICE}"        ]] || die 1 "--device is required"
    [[ -n "${WORKLOADS_CSV}" ]] || die 1 "--workloads is required"
    [[ -n "${RUNS}"          ]] || die 1 "--runs is required"
    [[ -n "${OUT_DIR}"       ]] || die 1 "--out is required"
    [[ -n "${BUILD_ID}"      ]] || die 1 "--build-id is required"

    if ! [[ "${RUNS}" =~ ^[1-9][0-9]*$ ]]; then
        die 1 "--runs must be a positive integer (got: ${RUNS})"
    fi
}

validate_adb() {
    command -v adb >/dev/null 2>&1 || die 2 "adb not in PATH"
    local state
    state="$(adb -s "${DEVICE}" get-state 2>/dev/null || true)"
    if [[ "${state}" != "device" ]]; then
        die 2 "device ${DEVICE} not in 'device' state (got: '${state}')"
    fi
}

wait_for_boot() {
    local deadline=$(( $(date +%s) + REBOOT_TIMEOUT ))
    log "waiting for sys.boot_completed (timeout ${REBOOT_TIMEOUT}s)"
    while (( $(date +%s) < deadline )); do
        local v
        v="$(adb -s "${DEVICE}" shell getprop sys.boot_completed 2>/dev/null \
             | tr -d '\r\n' || true)"
        if [[ "${v}" == "1" ]]; then
            # Give services a couple of seconds to settle.
            sleep 3
            log "device booted"
            return 0
        fi
        sleep 2
    done
    die 3 "reboot wait timed out for ${DEVICE}"
}

reboot_device() {
    log "rebooting ${DEVICE}"
    adb -s "${DEVICE}" reboot
    # adb may return immediately; wait until daemon comes back.
    sleep 5
    adb -s "${DEVICE}" wait-for-device
    wait_for_boot
}

set_ml_arm() {
    local arm="$1"  # "on" or "off"
    local val
    case "${arm}" in
        on)  val="true"  ;;
        off) val="false" ;;
        *)   die 1 "internal: unknown arm '${arm}'" ;;
    esac
    log "setting ro.lmk.use_ml_predictor=${val} ro.lmk.ml_threshold=${ML_THRESHOLD}"
    adb -s "${DEVICE}" root >/dev/null 2>&1 || true
    adb -s "${DEVICE}" shell "setprop ro.lmk.use_ml_predictor ${val}"
    adb -s "${DEVICE}" shell "setprop ro.lmk.ml_threshold ${ML_THRESHOLD}"
    # Restart lmkd so the new property is picked up at startup.
    adb -s "${DEVICE}" shell 'stop lmkd; start lmkd' || true
    sleep 2
}

run_cell() {
    local workload="$1"
    local arm="$2"          # "on" / "off"
    local run_idx="$3"
    local run_id="${workload}_ml_${arm}_${run_idx}"
    local cell_dir="${OUT_DIR}/${run_id}"
    mkdir -p "${cell_dir}"

    local workload_sh="${WORKLOAD_DIR}/${workload}.sh"
    [[ -x "${workload_sh}" ]] || die 4 "missing workload script: ${workload_sh}"

    cat > "${cell_dir}/manifest.json" <<EOF
{
  "run_id": "${run_id}",
  "workload": "${workload}",
  "ml": "${arm}",
  "run_index": ${run_idx},
  "device": "${DEVICE}",
  "build_id": "${BUILD_ID}",
  "duration_sec": ${DURATION_SEC},
  "ml_threshold": ${ML_THRESHOLD},
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    reboot_device
    set_ml_arm "${arm}"

    # Start metrics collector in the background; it tails logcat and
    # samples /proc periodically for DURATION_SEC.
    log "starting collector for ${run_id}"
    "${COLLECT_SCRIPT}" \
        --device "${DEVICE}" \
        --out "${cell_dir}" \
        --duration "${DURATION_SEC}" \
        >"${cell_dir}/collect.stdout" 2>"${cell_dir}/collect.stderr" &
    local collect_pid=$!

    # Run the workload synchronously. The workload owns its own duration;
    # we pass the same DURATION_SEC so it stays inside the collection
    # window.
    log "running workload ${workload} (${DURATION_SEC}s)"
    if ! "${workload_sh}" --device "${DEVICE}" --duration "${DURATION_SEC}" \
            >"${cell_dir}/workload.stdout" 2>"${cell_dir}/workload.stderr"; then
        log "workload ${workload} returned non-zero — metrics may be partial"
    fi

    # Wait for collector to finish its window.
    if ! wait "${collect_pid}"; then
        log "collector failed for ${run_id}"
        echo "${run_id}" >> "${OUT_DIR}/.failed_cells"
    fi

    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${cell_dir}/finished_utc.txt"
    log "cell ${run_id} done"
}

main() {
    parse_args "$@"
    validate_adb
    mkdir -p "${OUT_DIR}"
    [[ -x "${COLLECT_SCRIPT}" ]] || die 4 "collect script not executable: ${COLLECT_SCRIPT}"

    # Persist run config for downstream aggregation.
    cat > "${OUT_DIR}/run_config.json" <<EOF
{
  "device": "${DEVICE}",
  "build_id": "${BUILD_ID}",
  "workloads": "${WORKLOADS_CSV}",
  "runs": ${RUNS},
  "duration_sec": ${DURATION_SEC},
  "ml_threshold": ${ML_THRESHOLD},
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    IFS=',' read -ra WL_LIST <<< "${WORKLOADS_CSV}"

    # Matrix: workload x run x arm. Reboot is performed inside run_cell
    # so every cell starts from a clean PSI history.
    for wl in "${WL_LIST[@]}"; do
        for ((i=1; i<=RUNS; i++)); do
            for arm in off on; do
                run_cell "${wl}" "${arm}" "${i}"
            done
        done
    done

    if [[ -s "${OUT_DIR}/.failed_cells" ]]; then
        log "some cells failed; see ${OUT_DIR}/.failed_cells"
        exit 5
    fi

    log "all cells completed; results under ${OUT_DIR}"
}

main "$@"
