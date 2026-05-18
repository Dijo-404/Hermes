# Phase 5 — A/B Benchmark Harness

Reproducible benchmark for the lmkd ML PSI predictor (Phase 4). All
scripts in this directory drive a single device (rooted Pixel 4a or
equivalent) and produce a Markdown verdict report against the success
gates from `plan-executable.md` §Phase 5 / `plan.md` §2.

```
research/bench/
  ab.sh                  # top-level harness
  collect_metrics.sh     # per-run telemetry collector (invoked by ab.sh)
  aggregate.py           # raw artifacts -> summary.csv
  analyze.py             # summary.csv  -> report.md + gate verdict
  workloads/
    idle.sh
    web_scroll.sh
    app_switch.sh
    camera_burst.sh
    game_load.sh         # requires BENCH_GAME_PKG env var
```

## Prerequisites

1. **Device.** Rooted Pixel 4a on AOSP `android-latest-release`, ADB
   reachable over USB (`adb devices` shows the serial). `adb root` must
   succeed; the harness calls `setprop` for `ro.lmk.use_ml_predictor`
   and `ro.lmk.ml_threshold`, both of which require root.
2. **Binary.** A single `lmkd` build with `LMKD_USE_ML=true` flashed
   to `/system/bin/lmkd`. The same binary is used in both A and B arms
   (per Phase 5 anti-pattern row 1, the baseline is the same build with
   the runtime flag flipped, **not** an older or differently-built
   lmkd).
3. **Model artifacts.** Push the ONNX model and normalization sidecar
   produced by Phase 3 to the device:
   ```
   adb push research/psi_predictor.onnx /system/etc/lmkd/psi_predictor.onnx
   adb push research/psi_norm.json      /system/etc/lmkd/psi_norm.json
   adb shell chmod 0644 /system/etc/lmkd/*
   ```
   Properties `ro.lmk.ml_model_path` and `ro.lmk.ml_norm_path` should
   point at the above (set in `/system/build.prop` or via
   `resetprop -n`).
4. **Host Python.** `pip install -r ../requirements.txt` — pulls
   `pandas` and `numpy` which the aggregator and analyzer need. No
   torch / onnxruntime needed on the host for Phase 5.
5. **Host bash.** Bash 4+, `adb` in `$PATH`. On Windows, run the
   harness under WSL or Git Bash; PowerShell will not execute the
   bash scripts directly.

## Running the full matrix

```
chmod +x research/bench/ab.sh research/bench/collect_metrics.sh
chmod +x research/bench/workloads/*.sh

# Optional: package name for the high-memory workload.
export BENCH_GAME_PKG=com.example.large.game

research/bench/ab.sh \
    --device <serial> \
    --workloads idle,web_scroll,app_switch,camera_burst,game_load \
    --runs 5 \
    --out ./results/$(date -u +%Y%m%dT%H%M%SZ) \
    --build-id "$(git rev-parse --short HEAD)"
```

Then aggregate and analyze:

```
python research/bench/aggregate.py \
    --in 'results/<run-stamp>/*' \
    --out results/<run-stamp>/summary.csv

python research/bench/analyze.py \
    --summary results/<run-stamp>/summary.csv \
    --out     results/<run-stamp>/report.md
```

`analyze.py` exit codes:
- `0` aggregate verdict PASS.
- `2` summary CSV missing/empty.
- `3` aggregate verdict FAIL (report still written).

## What each cell does

Per `(workload, run-index, ml ∈ {on, off})` cell:

1. `adb reboot`, then poll `getprop sys.boot_completed` until `1`
   (90s timeout — Phase 5 anti-pattern row 2: PSI state carries
   across, so we reboot between every cell, not just between
   workloads).
2. `setprop ro.lmk.use_ml_predictor true|false` and
   `setprop ro.lmk.ml_threshold ${BENCH_ML_THRESHOLD:-0.65}`.
   `stop lmkd; start lmkd` to pick up the toggle at startup.
3. Run `coldstart_pre.txt` probes (5 cold starts via `am start -W`).
4. Start `logcat -s lmkd:I lmkd-ml:I` redirecting into `lmkd.log`.
   This captures both:
   - `Kill 'name' (pid), uid X, ...` lines from
     `lmkd.cpp:2506/2513` (counted as kill events).
   - `lmkd-ml: pre-emptive kill triggered (p=...)` from
     `ml_predictor.cpp` and `inference latency p50=NN us p99=MM us
     (n=...)` rolling-window emissions every 10 s.
5. Every 30 s during the run, snapshot
   `/proc/$(pidof lmkd)/status` (for `VmRSS` delta) and
   `dumpsys gfxinfo <pkg>` for each `BENCH_GFX_PKG` package
   (default `com.android.systemui,com.android.chrome`).
6. After the workload duration elapses, run `coldstart_post.txt`
   probes.

Run-id format (also the cell directory name): `<workload>_ml_<on|off>_<run>`.

## Metrics and success gates

`analyze.py` evaluates four gates per workload. The constants are at
the top of the script:

| Gate | Constant | Source |
|---|---|---|
| `jank_delta_pct  <= -30%` | `JANK_DELTA_PCT_MAX  = -30.0` | plan.md §2 |
| `kills_delta_pct <= +5%`  | `KILLS_DELTA_PCT_MAX = +5.0`  | plan.md §2 |
| `inf_p99_ms      <= 2.0`  | `INF_P99_MS_MAX      = 2.0`   | plan.md §Phase 3/4 |
| `vmrss_delta_kb  <= 4096` | `VMRSS_DELTA_KB_MAX  = 4096`  | plan-executable.md §Phase 5 |

Bootstrap 95% CIs (N=1000 resamples, seed pinned for reproducibility)
are reported on both the jank-percent delta and the kills-per-hour
delta. The aggregate verdict is PASS only when every workload clears
every gate.

## Expected runtime

Per cell: 90s reboot + ~10 min workload + ~30 s cold-start probes ≈
12 minutes. Full matrix:

```
5 workloads x 5 runs x 2 arms x ~12 min = ~5 hours
```

If you trim to `--runs 3`, ~3 hours. The aggregate/analysis phase is
sub-second on the host.

## Pitfalls

- **Anti-pattern row 1 — different binaries.** Do **not** compare the
  ML-enabled lmkd against an older or unmodified lmkd. Build once with
  `LMKD_USE_ML=true`, flash once, flip `ro.lmk.use_ml_predictor`
  between cells.
- **Anti-pattern row 2 — same boot.** Do **not** run the on and off
  arms in the same boot. PSI counters and lmkd's internal rolling
  state carry across; the harness enforces a reboot, do not bypass it.
- **Background sync.** Disable Google account sync / Play auto-updates
  before the run; background downloads inject memory pressure that
  pollutes the comparison.
- **Charging state.** Keep the device plugged in. Thermal throttling
  shifts cold-start times and confounds the inference-latency gate.
- **`ro.lmk.*` cached.** Some Android builds cache `ro.` properties at
  init. If the toggle doesn't take effect, switch to `persist.lmk.*`
  aliases (`ml_predictor.cpp` reads `ro.lmk.use_ml_predictor`; verify
  the property actually flipped via `getprop` between cells).
- **Camera permissions.** `camera_burst.sh` assumes the camera app is
  pre-permissioned. First-run permission dialogs will block the
  workload silently.
