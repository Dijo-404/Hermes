# Hermes — ML-Driven Predictive PSI Tuning for `lmkd`

> Research fork of `platform/system/memory/lmkd` (AOSP) that augments the
> Low Memory Killer Daemon's static PSI threshold logic with an optional,
> on-device LSTM that predicts impending memory-pressure kills 200–500 ms
> ahead of time. The ML path is **off by default** and gated at both build
> time (`LMKD_USE_ML` cflag, off in the upstream `Android.bp` defaults) and
> runtime (`persist.lmk.use_ml_predictor=false` by default), so an
> unmodified AOSP build that picks up these sources stays byte-for-byte
> behavior-compatible with stock `lmkd`.

## Title & Summary

`lmkd` ships with hard-coded PSI thresholds (`ro.lmk.psi_partial_stall_ms`,
`ro.lmk.psi_complete_stall_ms`) that act *reactively* — by the time a
threshold trips, the foreground app has typically already dropped frames.
This project replaces nothing; it inserts a small **predictive shim**
inside `__mp_event_psi` that pushes each PSI sample into a 20-step rolling
window and asks an ONNX-Runtime LSTM whether a kill is likely in the next
~200–500 ms. If the model fires (default threshold `0.65`), `lmkd` calls
the existing `find_and_kill_process` early; if it does not fire, control
falls through to the unchanged static-threshold decision tree.

The target success bar, taken from
[plan.md §2 "Research Hypothesis"](plan.md):

- **≥ 30 %** reduction in UI jank events (dropped frames via `dumpsys gfxinfo`).
- **≤ 5 %** increase in background kill frequency vs. static baseline.
- **≤ 2 ms** model inference latency on-device (mandatory — `lmkd` is
  latency-critical).
- **≤ 4 MB** RSS overhead from the inference engine.

This artifact contains the full source change, the dataset-collection
and training pipeline, the on-device A/B harness, and reproduction
instructions. **Real device numbers are not bundled** — the AOSP build
environment, a rooted Pixel-class device, and on-device training data
are all required to run the bench. Section *Reproduction Steps* below
walks an AOSP engineer through producing them; section *Limitations*
documents what this artifact does and does not prove.

## Repo Layout

```
Hermes/
├── lmkd.cpp                       upstream daemon + gated ML hook (see below)
├── ml_predictor.{h,cpp}           ONNX-Runtime PSIPredictor (LMKD_USE_ML only)
├── Android.bp                     adds `lmkd_ml_defaults` cc_defaults block
│                                    (enabled:false by default; flips with
│                                    SOONG_CONFIG_lmkd_use_hooks-style override
│                                    or a downstream patch flipping
│                                    `enabled: true`)
├── lmkd.rc, reaper.{cpp,h},       unchanged from upstream
│   watchdog.{cpp,h}, statslog.*,
│   libpsi/, include/, tests/
├── plan.md                        original 10-week project plan
├── plan-executable.md             phase-by-phase, LLM-friendly executable
│                                    version with verified file:line anchors
└── research/                      training + benchmarking artifacts
    ├── README.md                  research-side reproduction notes
    ├── requirements.txt           pinned PyTorch / onnxruntime versions
    ├── notes/
    │   ├── phase1-epoll-wiring.md PSI fd → mp_event_psi dispatch trace
    │   └── phase1-callgraph.md    full PSI-event-to-reap call graph
    ├── collector.py               on-device PSI/meminfo/foreground-RSS sampler
    ├── label.py                   logcat-driven kill labeler (T-200 to T-100 ms)
    ├── dataset.py                 windowed tensor builder + NormStats
    ├── eda.ipynb                  class-balance / per-scenario sanity checks
    ├── model.py                   LSTM(6→32→1) ≤200 K params
    ├── train.py                   leave-one-scenario-out training driver
    ├── export_onnx.py             opset_version=11 exporter, PyTorch parity check
    ├── bench_onnx.py              CPU p99 inference latency harness
    ├── model_card.md              metrics, training-data hash, code commit
    └── bench/
        ├── README.md              A/B harness operator manual
        ├── ab.sh                  flashed-once, prop-flipped run loop
        ├── collect_metrics.sh     dumpsys gfxinfo / lmkd kill / RSS scraper
        ├── aggregate.py           per-cell CSV aggregator
        ├── analyze.py             paired bootstrap 95 % CI
        └── workloads/             per-scenario adb scripts
```

### Where the ML hook lives in `lmkd.cpp`

The injection point — verified in Phase 1 and re-confirmed against this
tree — is inside `__mp_event_psi`, at **`lmkd.cpp:2936-2989`** (the
block bracketed by `#ifdef LMKD_USE_ML`). The hook sits *before* the
static-threshold decision tree, so:

- when the ML build flag is off, the file compiles identically to
  upstream;
- when the flag is on but `persist.lmk.use_ml_predictor=false`,
  `PSIPredictor::instance()` returns `nullptr` and the `if` predicate
  short-circuits with zero allocations and ~one branch of overhead;
- when both are on but the rolling window is < 20 samples or
  `predict()` < threshold, control falls through to the unchanged
  static path.

The PSI dispatch chain that delivers events to this hook is documented
fully in `research/notes/phase1-callgraph.md` — kernel raises `EPOLLPRI`
→ main `epoll_wait` loop (`lmkd.cpp:3980` / `3995` / `4007`) →
`call_handler` (`lmkd.cpp:3965`) → `mp_event_psi` (`lmkd.cpp:3173`,
the thin shim that delegates at 3175) → `__mp_event_psi`
(`lmkd.cpp:2717`, where the ML hook at 2936 sits) →
`find_and_kill_process` (`lmkd.cpp:2539`).

## Reproduction Steps

Five phases mapped 1:1 to the corresponding sections of
[plan-executable.md](plan-executable.md). Times below assume a
mid-range x86 dev workstation with an AOSP checkout already synced and
a rooted Pixel-class device on USB.

### Step 1 — Build `lmkd` with the ML flag (Phase 4)

**Where:** AOSP tree containing this repo at
`system/memory/lmkd/`.

**What runs:**

```bash
# From the AOSP top-level
source build/envsetup.sh
lunch aosp_<your-device>-userdebug

# Flip the ML defaults to enabled. Two equivalent options:
#   (a) edit Android.bp's lmkd_ml_defaults to `enabled: true`, or
#   (b) overlay a downstream cc_defaults that sets the same fields.
m lmkd
```

**Expected output:** `out/target/product/<device>/system/bin/lmkd`
linked against `libonnxruntime`. Verify with
`readelf -d out/.../system/bin/lmkd | grep onnxruntime`.

Also build with the flag *off* to confirm the byte-compat property:

```bash
# Leave Android.bp's lmkd_ml_defaults at `enabled: false`
m lmkd
# Resulting binary must be identical to a tree without ml_predictor.cpp.
```

**Time:** ~5–20 min depending on ccache state.

### Step 2 — Collect a labeled dataset (Phase 2)

**Where:** developer host, device attached via `adb`.

**What runs:**

```bash
cd research/
pip install -r requirements.txt

# Per scenario from plan.md §Phase 2:
#   light-browse, heavy-tab-switch, camera-burst, game-cold-start, mixed.
# The collector samples every 100 ms; label.py post-processes the
# logcat capture and stamps kill_event=1 in the [T-200 ms, T-100 ms]
# window before each lmkd kill atom.
python collector.py --scenario heavy-tab-switch --device <serial> \
                    --duration 600 --out data/heavy-tab-switch-pixel4a.csv
python label.py    data/heavy-tab-switch-pixel4a.csv \
                   data/heavy-tab-switch-pixel4a.logcat
```

**Expected output:** `research/data/*.csv`, combined ≥ 50 000 rows
across ≥ 3 scenarios, positive-class fraction in `[0.005, 0.05]`.
Sanity-check with `research/eda.ipynb`.

**Time:** ~10 min per 10-minute scenario session; budget 1 day total
for a multi-scenario, multi-device corpus.

### Step 3 — Train and export (Phase 3)

**Where:** developer host (GPU optional — model is tiny).

**What runs:**

```bash
cd research/
python train.py --data data/*.csv --epochs 25 --leave-one-out
python export_onnx.py --checkpoint runs/best.pt \
                      --out psi_predictor.onnx \
                      --norm-out normalization.json
python bench_onnx.py psi_predictor.onnx        # CPU p99 must be ≤ 2 ms
```

**Expected output:**

- `research/runs/best.pt` (PyTorch checkpoint).
- `research/psi_predictor.onnx` (opset 11; parity-checked against
  PyTorch within 1e-5 on 100 random windows by `export_onnx.py`).
- `research/normalization.json` matching the `NormStats.to_json` schema
  consumed by `ml_predictor.cpp`.
- `research/model_card.md` updated with leave-one-scenario-out
  recall/precision and the lead-time histogram.

**Time:** ~15–45 min depending on epoch count and corpus size.

### Step 4 — Push artifacts to device (Phase 4 wiring)

**Where:** rooted device on `adb`.

**What runs:**

```bash
adb root && adb remount
adb push research/psi_predictor.onnx /system/etc/lmkd/psi_predictor.onnx
adb push research/normalization.json /system/etc/lmkd/normalization.json
adb shell setprop persist.lmk.ml_model_path /system/etc/lmkd/psi_predictor.onnx
adb shell setprop persist.lmk.ml_norm_path  /system/etc/lmkd/normalization.json
adb shell setprop persist.lmk.use_ml_predictor true
adb shell stop lmkd && adb shell start lmkd
adb logcat -s lmkd | grep -E "lmkd-ml|PSIPredictor"
```

**Expected output:** A `PSIPredictor loaded model` line, periodic
`lmkd-ml: latency p50=…us p99=…us` lines (the rolling-window report
plumbed in Phase 4), and — once the device sees pressure —
`lmkd-ml: pre-emptive kill triggered (p=0.xxx)` lines from
`lmkd.cpp:2964`.

**Time:** seconds.

### Step 5 — Benchmark and analyze (Phase 5)

**Where:** rooted device on `adb`, host running the harness.

**What runs:**

```bash
cd research/bench/
DEVICE=<serial> BENCH_ML_THRESHOLD=0.65 ./ab.sh
python aggregate.py results/<run-id>/
python analyze.py   results/<run-id>/summary.csv
```

`ab.sh` reboots the device between cells, flips
`persist.lmk.use_ml_predictor` between `true` and `false`, runs each
workload from `workloads/` for 10 minutes, and scrapes
`dumpsys gfxinfo`, lmkd kill atoms, lmkd VmRSS deltas, and
`am start -W` cold-start times into per-cell JSON.

**Expected output:** `research/results/<run-id>/summary.csv` with no
NaN cells; `analyze.py` prints paired bootstrap 95 % CIs for jank
delta and kill-frequency delta.

**Time:** ~10 min per cell × 2 conditions × 5 workloads × ≥ 5 reps
≈ 8 hours. Use `tmux` or a babysit script.

## Configuration

All runtime knobs are exposed as Android system properties (Phase 5
rename — `persist.` prefix means they survive reboot and are flippable
via `adb shell setprop`):

| Property | Type | Default | Purpose |
|---|---|---|---|
| `persist.lmk.use_ml_predictor` | bool | `false` | Master gate. When `false`, `PSIPredictor::instance()` returns `nullptr` and the entire ML branch in `lmkd.cpp:2936-2989` is dead. Must be `true` for any inference to occur. |
| `persist.lmk.ml_model_path` | string | `/system/etc/lmkd/psi_predictor.onnx` | Path to the exported ONNX model. Loaded lazily on the first PSI event so daemon startup is unaffected by missing files. |
| `persist.lmk.ml_norm_path` | string | `/system/etc/lmkd/normalization.json` | Path to the z-score normalization sidecar. Schema must match `research/dataset.py::NormStats.to_json` (`{"feature_order": […], "mean": […], "std": […]}`). |
| `persist.lmk.ml_threshold` | float | `0.65` | Sigmoid cutoff for "fire a pre-emptive kill". Lower = more aggressive (more kills, fewer jank events); higher = more conservative. Tune per device after collecting a baseline CSV. |

Property names are sourced from
[`ml_predictor.cpp:38-41`](ml_predictor.cpp). The Phase 5 commit
`685bfea` renamed these from `ro.lmk.*` to `persist.lmk.*` so they can
be flipped at runtime without reflashing.

## Architecture Diagram

Text-only call graph from kernel PSI event to reap, with the ML hook
called out. This is a condensed re-rendering of
[`research/notes/phase1-callgraph.md`](research/notes/phase1-callgraph.md);
see that file for the full anchor-by-anchor walk.

```
                ┌───────────────────────────────────────────────────┐
                │ kernel: PSI threshold breached on memory pressure │
                │   → raises EPOLLPRI on the per-level PSI fd       │
                │     (fd opened by init_psi_monitor,               │
                │      libpsi/psi.cpp:36)                           │
                └───────────────────────┬───────────────────────────┘
                                        │
                            EPOLLPRI on PSI fd
                                        │
                                        ▼
         ┌─────────────────────────────────────────────────────────┐
         │ main loop: epoll_wait                                   │
         │   lmkd.cpp:3980 (timed) / 3995 (kill-timeout) /         │
         │   4007 (blocking idle).  Second-pass scan at 4040 picks │
         │   the non-HUP event; evt->data.ptr resolves to the      │
         │   event_handler_info* stashed by register_psi_monitor   │
         │   (libpsi/psi.cpp:91).                                  │
         └─────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────────┐
                │ call_handler (lmkd.cpp:3965)        │
                │   handler_info->handler(...)        │
                │   resolves to mp_event_psi          │
                └─────────────────────┬───────────────┘
                                      │
                                      ▼
              ┌──────────────────────────────────────────┐
              │ mp_event_psi  (lmkd.cpp:3117 — shim)     │
              │   packs level → psi_event_data and       │
              │   delegates to __mp_event_psi at 3119    │
              └──────────────────────┬───────────────────┘
                                     │
                                     ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ __mp_event_psi  (lmkd.cpp:2717 — kill-decision body)          │
   │                                                               │
   │   reads PSI/meminfo/vmstat, computes thrashing, picks         │
   │   min_score_adj …                                             │
   │                                                               │
   │   ┌─────────────────────────────────────────────────────────┐ │
   │   │ #ifdef LMKD_USE_ML  (lmkd.cpp:2936-2989)                │ │
   │   │                                                         │ │
   │   │   if (PSIPredictor* ml = PSIPredictor::instance()) {    │ │
   │   │     ml->push_sample(some_avg10, some_avg60, some_total, │ │
   │   │                     full_avg10, full_total,             │ │
   │   │                     mem_avail_kb_approx);               │ │
   │   │     if (ml->ready() &&                                  │ │
   │   │         ml->predict() >= ml->threshold()) {             │ │
   │   │       ALOGI("lmkd-ml: pre-emptive kill …");             │ │
   │   │       find_and_kill_process(...);   // EARLY FIRE       │ │
   │   │       goto no_kill;                                     │ │
   │   │     }                                                   │ │
   │   │   }                                                     │ │
   │   │   // else: fall through to static-threshold path        │ │
   │   │ #endif                                                  │ │
   │   └─────────────────────────────────────────────────────────┘ │
   │                                                               │
   │   …existing static decision tree (unchanged)…                 │
   │   → find_and_kill_process(lmkd.cpp:2539)                      │
   └──────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │ find_and_kill_process (lmkd.cpp:2539)│
               │   iterates oom_score_adj buckets,    │
               │   picks victim, calls                │
               │   kill_one_process (lmkd.cpp:2422)   │
               │   → reaper.kill(...) (2483)          │
               │   → reaper thread pidfd_send_signal +│
               │     process_mrelease (reaper.cpp)    │
               └──────────────────────────────────────┘
```

## Limitations & Known Gaps

This artifact is a code-complete research scaffold; it is **not** a
device-verified result. Treat every quantitative claim below as
"to be measured by the reproducer."

- **No on-device numbers in-tree.** No AOSP build environment or
  rooted device was available during this artifact's construction.
  Bench p50/p99 latency, jank deltas, kill-frequency deltas, and RSS
  overhead are all `<TBD>` — the harness in `research/bench/` is the
  vehicle for producing them. The plan.md success bars (≥ 30 % jank
  reduction, ≤ 5 % kill-freq increase, ≤ 2 ms inference, ≤ 4 MB RSS)
  remain the *target*, not a *result*.

- **`mem_available_kb` is approximated.** The Phase 4 code review
  flagged this: `lmkd.cpp:2958` feeds the predictor
  `mi.field.easy_available * page_k` (free + inactive_file in pages,
  converted to KB), not `/proc/meminfo`'s `MemAvailable`. The model
  was trained on the latter (via `collector.py`'s `/proc/meminfo`
  parse). Divergence is absorbed by z-score normalization in practice,
  but the residual is unmeasured. A clean fix is to parse
  `MemAvailable` once per PSI event; we left it for a follow-up to
  keep the upstream diff narrow.

- **Single-device training risk** (plan.md §Risk Register row 4). If
  the dataset in Step 2 is collected on a single device class, the
  resulting model is likely to overfit that device's PSI envelope.
  Reproducers must collect across at least two device classes before
  shipping the threshold value.

- **JSON parser in `ml_predictor.cpp` assumes a flat top-level array
  schema** (Phase 4 code review). The parser was deliberately written
  without a full JSON dependency to keep the lmkd binary slim; it
  tolerates the exact format emitted by `research/dataset.py::NormStats.to_json`
  and will fail loudly on anything else. If you hand-edit the
  normalization file, keep the `mean` and `std` arrays as flat lists
  of six floats.

- **Static-threshold path is preserved, not removed.** This is by
  design (Phase 4 anti-pattern guard) — the model augments, it does
  not replace. If the model returns `predict() < threshold`, the
  existing static tree runs unchanged.

- **`mp_event_psi` handler-installation is conditional on
  `use_new_strategy`** (set at `lmkd.cpp:3625`). When that flag is false,
  the PSI epoll slot routes to `mp_event_common` instead and the ML
  hook is never reached. Verify `use_new_strategy=true` on your
  target device before benching.

## References

- [plan.md](plan.md) — original 10-week project plan (problem
  statement, hypothesis, success metrics, risk register).
- [plan-executable.md](plan-executable.md) — phase-by-phase executable
  plan with verified file:line anchors. Phase 0 is the load-bearing
  reference card.
- [research/notes/phase1-epoll-wiring.md](research/notes/phase1-epoll-wiring.md)
  — exact `epoll_ctl` site that wires PSI fd → `mp_event_psi`.
- [research/notes/phase1-callgraph.md](research/notes/phase1-callgraph.md)
  — full kernel-event → reaper call graph.
- [research/model_card.md](research/model_card.md) — model architecture,
  parameter count, training data hash, recall/precision per scenario.
- [research/bench/README.md](research/bench/README.md) — operator
  manual for the A/B harness.
- AOSP `lmkd` upstream:
  [`platform/system/memory/lmkd`](https://android.googlesource.com/platform/system/memory/lmkd/+/refs/heads/main).
- AOSP `libpsi`: same tree, `libpsi/` subdir.
- Linux PSI kernel documentation:
  [`Documentation/accounting/psi.rst`](https://www.kernel.org/doc/html/latest/accounting/psi.html).
- ONNX Runtime Android: [onnxruntime.ai/docs/install/#install-on-android](https://onnxruntime.ai/docs/install/#install-on-android).
