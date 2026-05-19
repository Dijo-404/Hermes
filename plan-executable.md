# plan-executable.md — ML-Driven Predictive PSI Tuning for lmkd

> LLM-friendly, phase-by-phase executable version of [plan.md](plan.md).
> Each phase is self-contained: a new chat can pick up any phase using only that
> phase's section + the Phase 0 reference card.
>
> Source plan: research/architectural document. This document: copy-from-source
> instructions, verified file:line references, verification checklists,
> anti-pattern guards.

---

## Phase 0 — Documentation Discovery (Reference Card)

**Purpose:** Ground-truth map of the lmkd codebase. Every subsequent phase
references this card instead of re-deriving facts.

### Allowed APIs / Verified Anchors

| Symbol | Location | Signature | Role |
|---|---|---|---|
| `mp_event_psi` | [lmkd.cpp:3117](lmkd.cpp#L3117) | `static void mp_event_psi(int data, uint32_t events, struct polling_params *poll_params)` | Epoll callback; thin wrapper that delegates to `__mp_event_psi`. **Injection point candidate A.** |
| `__mp_event_psi` | [lmkd.cpp](lmkd.cpp) (call site at 3117–3120) | internal | Actual PSI-event handling body. **Injection point candidate B (preferred — has full pressure context).** |
| `find_and_kill_process` | [lmkd.cpp:2539](lmkd.cpp#L2539) | `static int find_and_kill_process(int min_score_adj, struct kill_info *ki, union meminfo *mi, struct wakeup_info *wi, struct timespec *tm, struct psi_data *pd)` | Iterates OOM-adj buckets, selects victim, invokes `kill_one_process`. |
| `kill_one_process` | [lmkd.cpp](lmkd.cpp) (called from `find_and_kill_process`) | internal | Single-process kill executor; eventually triggers reaper. |
| `init_psi_monitor` | [libpsi/psi.cpp:36](libpsi/psi.cpp#L36) | `int init_psi_monitor(enum psi_stall_type stall_type, int threshold_us, int window_us, enum psi_resource resource)` | Opens `/proc/pressure/<res>`, writes kernel subscription, returns FD. |
| `parse_psi_line` | [libpsi/psi.cpp:109](libpsi/psi.cpp#L109) | internal parser | Parses `some avg10=… avg60=… avg300=… total=…`. |
| `poll_kernel` | [lmkd.cpp:846](lmkd.cpp#L846) | `static void poll_kernel(int poll_fd)` | **NOT the main epoll loop** (plan.md error). Actually reads the eBPF kill-outcome ring buffer. The PSI epoll dispatch lives elsewhere — locate it before Phase 4 (see Phase 1 task). |
| `init_psi_monitors` | [lmkd.cpp:3579](lmkd.cpp#L3579) | applies `psi_partial_stall_ms` and `psi_complete_stall_ms` to kernel thresholds | The static-threshold seam — model output can replace these values dynamically. |
| `lmkd_pack_set_kill_occurred` | [statslog.cpp:298](statslog.cpp#L298) | packs `struct kill_stat` for atom logging | Source of training labels. |

### Verified Data Structures

| Struct | File:Line | Fields |
|---|---|---|
| `psi_stats` | [libpsi/include/psi/psi.h:33](libpsi/include/psi/psi.h#L33) | `float avg10; float avg60; float avg300; unsigned long total;` |
| `psi_data` (container) | [libpsi/include/psi/psi.h:40](libpsi/include/psi/psi.h#L40) | `mem_stats[2] /* SOME, FULL */, io_stats[2], cpu_stats[2]` |
| `enum vmpressure_level` | [lmkd.cpp:181](lmkd.cpp#L181) | `LOW=0, MEDIUM, CRITICAL, COUNT` (internal — **not in lmkd.h**) |
| `level_oomadj[]` | [lmkd.cpp:209](lmkd.cpp#L209) | `static int level_oomadj[VMPRESS_LEVEL_COUNT];` |
| `psi_partial_stall_ms` / `psi_complete_stall_ms` | [lmkd.cpp:231](lmkd.cpp#L231), read at [lmkd.cpp:4153](lmkd.cpp#L4153) | static ints, loaded via `GET_LMK_PROPERTY` |
| `struct kill_stat` | [statslog.h:77](statslog.h#L77) | `uid, taskname, kill_reason, oom_score, min_oom_score, free_mem_kb, free_swap_kb, thrashing, max_thrashing` |
| `struct memory_stat` | [statslog.h:48](statslog.h#L48) | `pgfault, pgmajfault, rss_in_bytes, cache_in_bytes, swap_in_bytes, process_start_time_ns` |

### Plan.md Errors Corrected

1. **lmkd.h does NOT contain `oom_adj` structs or pressure-level enums.** Those
   are static internal types inside [lmkd.cpp](lmkd.cpp). lmkd.h only carries the
   on-the-wire LMK protocol packets (`LMK_TARGET`, `LMK_PROCPRIO`, …). Do not
   include lmkd.h to access `vmpressure_level` — declare any new type next to
   the existing ones in lmkd.cpp.
2. **`poll_kernel()` is not the main epoll loop.** Plan.md task "understand
   threading model" via poll_kernel is misdirected. The PSI epoll dispatch
   lives in lmkd's main loop (separate function). Phase 1 explicitly locates
   it.
3. **`kill_stat` has no standalone timestamp field.** Plan.md's "post-process
   CSV — mark kill_event=1 at T-200ms before each kill" must use the **logcat
   timestamp** of the statsd write, not a field inside the atom.
4. **`Android.bp` currently has zero ML/inference deps.** Phase 4's
   `libonnxruntime` addition is net-new — there is no pre-existing inference
   plumbing to extend.

### Anti-Patterns (Do Not Do)

- Do not invent fields on `struct kill_stat` (no `oom_score_adj`, no
  `timestamp` member exists today).
- Do not modify `include/lmkd.h` to add new internal enums — it is the public
  IPC header.
- Do not call inference from inside `mp_event_psi` (the wrapper at 3117) —
  prefer `__mp_event_psi` where the resolved pressure level is already in scope.
- Do not block in `predict()`: lmkd is latency-critical; budget 2ms p99.
- Do not link `libonnxruntime` without a fallback `cc_defaults` toggle — AOSP
  builds without it must still produce a working `lmkd`.

### Confidence + Gaps

- **High confidence:** All function locations, all data-structure fields,
  Android.bp dep list.
- **Gap (Phase 1 must close):** Exact line of the main epoll dispatch that
  registers `mp_event_psi` as a handler (PSI fd → callback wiring). Phase 0
  located the property write but not the dispatch.
- **Gap (Phase 2 must close):** Whether `statslog` kill atoms can be
  subscribed-to in real time from userspace, or only via post-hoc logcat
  parsing.

---

## Phase 1 — Environment & Codebase Familiarization

**Duration:** ~1 week.
**Self-contained context:** Phase 0 reference card above.

### Tasks

1. Build lmkd in the AOSP tree at `android-latest-release`. Confirm the
   binary at `out/.../lmkd` is produced from the files in this repo unchanged.
2. **Close the Phase 0 gap:** locate the PSI-fd-to-callback wiring. Search
   pattern (read with Grep, do not paste here):
   `mp_event_psi` as a function-pointer reference in [lmkd.cpp](lmkd.cpp), and
   the corresponding `epoll_ctl(EPOLL_CTL_ADD, …)` / `register_psi_monitor`
   call site. Record the file:line in `research/notes/phase1-epoll-wiring.md`.
3. Annotate the full PSI lifecycle from kernel-fd readiness → `__mp_event_psi`
   → `find_and_kill_process` → `kill_one_process` → reaper. Produce a call
   graph at `research/notes/phase1-callgraph.md`. Use the line anchors from
   the Phase 0 card; do not re-discover.
4. On a rooted Pixel 4a (or emulator with `-memory 2048`), exercise PSI by
   lowering `ro.lmk.psi_partial_stall_ms` to 10 and running
   `stress-ng --vm 2 --vm-bytes 80% --timeout 60s`. Capture
   `logcat -s lmkd` to `research/notes/phase1-stress-log.txt`.

### Verification

- [ ] `lmkd` binary builds from unmodified repo.
- [ ] `research/notes/phase1-epoll-wiring.md` exists and cites a `file:line`
  for the PSI fd registration (one of the Phase 0 gaps).
- [ ] Call graph references only verified symbols from the Phase 0 card.
- [ ] Stress log shows at least one lmkd-driven kill at the lowered threshold.

### Anti-Pattern Guards

- Don't assume `poll_kernel` is the PSI dispatcher. Phase 0 corrected this.
- Don't read pressure thresholds from `include/lmkd.h` — they aren't there.

---

## Phase 2 — Dataset Collection Pipeline

**Duration:** ~2 weeks.
**Self-contained context:** Phase 0 reference card + Phase 1 call graph.

### Tasks

1. Create `research/collector.py` that samples every 100 ms from a connected
   ADB device:
   - `/proc/pressure/memory` — parse with the same grammar as
     [libpsi/psi.cpp:109](libpsi/psi.cpp#L109) (`some avg10=… avg60=… avg300=…
     total=…`). **Copy the parse format from there**, do not reinvent.
   - `/proc/meminfo` — `MemAvailable`, `SwapFree`, `SwapTotal`.
   - Foreground app RSS via `/proc/<pid>/status` (resolve pid via
     `dumpsys activity activities`).
   - `oom_score_adj` distribution: `/proc/<pid>/oom_score_adj` for the top
     N processes (N=20).
2. Capture kill events via `logcat -b main -s lmkd:I` streamed alongside
   sampling. The kill log line emitted from
   [lmkd_pack_set_kill_occurred](statslog.cpp#L298) is the label source.
   **Use the logcat timestamp** as the kill instant — there is no field on
   `kill_stat` for this (Phase 0 correction).
3. Label rule: for each kill at time `T_k`, set `kill_event=1` on all rows in
   `[T_k - 200 ms, T_k - 100 ms]`. Everything else is 0.
4. Record sessions across the five workload scenarios listed in
   [plan.md §Phase 2](plan.md). Each session produces a CSV under
   `research/data/<scenario>-<device>-<timestamp>.csv`.
5. Run a quick EDA notebook (`research/eda.ipynb`) confirming:
   class balance (positives should be ~0.5–2 % of rows), no
   missing values, PSI avg10 distribution per scenario.

### Verification

- [ ] Combined CSV ≥ 50 000 rows across ≥ 3 scenarios.
- [ ] Positive-class fraction in [0.005, 0.05]. If outside, revisit label
  window or workload selection.
- [ ] EDA notebook executes top-to-bottom on a clean kernel.
- [ ] Parse code in `collector.py` round-trips against a sample
  `/proc/pressure/memory` snapshot byte-for-byte against
  `libpsi/psi.cpp` expectations.

### Anti-Pattern Guards

- Do not parse kill events out of `dumpsys` — only `logcat -s lmkd` carries
  the verified atom.
- Do not invent an `oom_score_adj` field on `kill_stat`. The atom logs
  `oom_score` and `min_oom_score` only (Phase 0).
- Do not sample at < 100 ms intervals on-device — ADB shell roundtrip
  overhead will skew timing. If sub-100 ms needed, run collector as an
  on-device native binary (defer to Phase 5).

---

## Phase 3 — Model Design & Training

**Duration:** ~2 weeks.
**Self-contained context:** Phase 0 card + Phase 2 CSV schema.

### Tasks

1. Implement the LSTM described in [plan.md §Phase 3](plan.md):
   - Input: 20 timesteps × 6 features (some_avg10, some_avg60, some_total,
     full_avg10, full_total, mem_avail_kb).
   - Architecture: `LSTM(input=6, hidden=32, layers=1)` → Dropout(0.2)
     → Linear(32 → 1) → Sigmoid.
   - Target parameter count: ≤ 200 K.
2. Train with class-weighted BCE (positive weight ≈ 10).
3. Evaluate against held-out scenarios (leave-one-scenario-out):
   - Recall ≥ 0.85, Precision ≥ 0.70.
   - Lead-time histogram: ≥ 80 % of true positives fire ≥ 100 ms before
     the kill.
4. Export to ONNX with `opset_version=11`, validate with
   `onnxruntime.InferenceSession` in Python — output must match PyTorch
   within 1e-5.
5. Save artifact: `research/psi_predictor.onnx` + `model_card.md` with
   metrics, training data hash, git commit of training code.

### Verification

- [ ] `research/train.py --quick` reproduces metrics within ±2 % of the
  reported numbers from a fixed seed.
- [ ] Exported ONNX, when loaded in Python, returns identical logits to
  PyTorch on 100 random windows.
- [ ] Parameter count printed at end of training is ≤ 200 000.
- [ ] Inference benchmark in `research/bench_onnx.py` reports p99 ≤ 2 ms on
  a Pixel 4a CPU (or equivalent ARM64 dev board).

### Anti-Pattern Guards

- Do not switch frameworks mid-phase (TF→PyTorch→ONNX juggling burns time).
  Lock to PyTorch → ONNX.
- Do not use `opset_version` > 11 without verifying ONNX Runtime Android
  support — older opsets are safer for the Android prebuilts.
- Do not train on data from a single device — Phase 2 must produce
  multi-device CSVs (Risk Register row 4 in plan.md).

---

## Phase 4 — C++ Inference Integration in lmkd

**Duration:** ~2 weeks.
**Self-contained context:** Phase 0 card + Phase 1 epoll wiring file.

### Tasks

1. Create `ml_predictor.h` and `ml_predictor.cpp` at repo root. Class shape
   matches [plan.md §Phase 4](plan.md). Public surface:
   - `PSIPredictor(const char* model_path);`
   - `void push_sample(...)` — 6 floats matching Phase 3 feature order.
   - `float predict();` — returns sigmoid output in [0,1].
2. Modify [Android.bp](Android.bp):
   - Add `cc_defaults { name: "lmkd_ml_defaults", … }` gated by a build flag
     `LMKD_USE_ML` so builds without onnxruntime still produce a working
     binary.
   - Inside the flag: `shared_libs: ["libonnxruntime"]`.
   - Source `ml_predictor.cpp` only when flag is on.
3. Wire `__mp_event_psi` (NOT `mp_event_psi` — see Phase 0 anti-patterns) to:
   - Push the current PSI sample (read in the same handler).
   - Call `g_predictor->predict()`.
   - If `prob ≥ KILL_THRESHOLD` (0.65 default, runtime-tunable via property
     `ro.lmk.ml_threshold`), invoke the existing
     [find_and_kill_process](lmkd.cpp#L2539) with the same args the static
     path would have used.
   - If below threshold, **fall through to the existing static path** —
     model is augmentation, not replacement.
4. Add a property `ro.lmk.use_ml_predictor` (default `false`) that gates
   model initialization in lmkd's startup path. When false, the binary is
   byte-for-byte behavior-compatible with the unmodified daemon.
5. Bracket every `predict()` call with `clock_gettime(CLOCK_MONOTONIC, …)`
   and log p50/p99 every 10 s via `ALOGI`.

### Verification

- [ ] `m lmkd` builds with `LMKD_USE_ML=true` and `LMKD_USE_ML=false`.
- [ ] With `ro.lmk.use_ml_predictor=false`, `lmkd` startup logs are
  byte-identical to the baseline (diff `logcat -s lmkd:V` before/after).
- [ ] With predictor on, `logcat -s lmkd:I | grep ml-` shows pre-emptive
  kill log lines.
- [ ] `clock_gettime` p99 from rolling window ≤ 2 ms over a 10-minute
  stress run.
- [ ] Grep guards (must return zero matches):
  - `grep -n "oom_score_adj" statslog.h` — invented field check.
  - `grep -n "vmpressure_level" include/lmkd.h` — wrong-header check.
  - `grep -n "libonnxruntime" Android.bp` outside the gated `cc_defaults` —
    leak check.

### Anti-Pattern Guards

- Do not put inference inside the wrapper `mp_event_psi` at
  [lmkd.cpp:3117](lmkd.cpp#L3117); the pressure level isn't resolved there.
  Use `__mp_event_psi`.
- Do not allocate per-call inside `predict()` — preallocate the rolling
  window deque and the ONNX input tensor at construction time.
- Do not remove the static-threshold path. Keep it as fallback under the
  model output (`else` branch in step 3.4).
- Do not block lmkd startup on model load — load lazily on first PSI event,
  or asynchronously on a worker thread.

---

## Phase 5 — Testing & Benchmarking

**Duration:** ~2 weeks.
**Self-contained context:** Phase 0 card + Phase 4 toggles
(`ro.lmk.use_ml_predictor`, `ro.lmk.ml_threshold`).

### Tasks

1. Reproducible A/B harness (`research/bench/ab.sh`) that:
   - Flashes the same lmkd binary on the device.
   - Switches `ro.lmk.use_ml_predictor` between runs.
   - Runs each workload from Phase 2 for 10 minutes.
   - Collects: `dumpsys gfxinfo <pkg>`, lmkd kill count from
     [statslog.cpp:298](statslog.cpp#L298) hits, p50/p99 inference latency
     from Phase 4 logging, `/proc/<lmkd-pid>/status` VmRSS delta,
     `am start -W <pkg>` cold-start times.
2. Aggregate into `research/results/<run-id>/summary.csv`. Each run keyed
   by `(workload, ml_on/off, device, build_id)`.
3. Statistical comparison: bootstrap 95 % CI for jank-frame delta. Pre-set
   success bar from [plan.md §2](plan.md) — ≥ 30 % jank reduction, ≤ 5 %
   kill-frequency increase.

### Verification

- [ ] Bench harness runs end-to-end with one command on a clean device.
- [ ] At least 5 runs per cell of the A/B matrix.
- [ ] Results CSV has no NaN/empty cells for primary metrics.
- [ ] Final report in `research/results/summary.md` includes CI intervals,
  not just point estimates.

### Anti-Pattern Guards

- Do not compare against an older lmkd binary — baseline must be the same
  build with `ro.lmk.use_ml_predictor=false`. Anything else conflates ML
  benefit with unrelated daemon changes.
- Do not run baseline and experimental in the same workload session — PSI
  state carries across; reboot between cells.

---

## Phase 6 — Write-Up & Upstream Patch

**Duration:** ~2 weeks.
**Self-contained context:** Phase 5 results CSV + summary.md.

### Tasks

1. Paper draft (LaTeX) per [plan.md §Phase 6](plan.md) outline.
2. Gerrit patch series — three changes:
   - `lmkd: introduce optional ML predictor scaffolding (no-op when disabled)`
   - `lmkd: add PSI prediction injection point in __mp_event_psi`
   - `lmkd: add ro.lmk.use_ml_predictor runtime toggle and bench logging`
3. `README_research.md` documenting full reproduction from `git clone` to
   final benchmark numbers.

### Verification

- [ ] Patch series applies cleanly on `android-latest-release` HEAD.
- [ ] `repo upload` succeeds; Gerrit Change-Id present on each commit.
- [ ] Reproduction README walked through by a fresh collaborator in
  ≤ 1 day on a Pixel 4a.

### Anti-Pattern Guards

- Do not bundle all three patches into one commit — AOSP review prefers
  small, reviewable changes.
- Do not omit the `Test:` and `Bug:` trailers — Gerrit will reject.
- Do not include `research/data/*.csv` in the upstream patch — keep
  research artifacts in the fork only.

---

## Final Verification Phase

After Phase 6, run these grep checks across the modified tree (must all
return zero matches):

- `grep -rn "oom_score_adj" statslog.h statslog.cpp` — invented-field guard.
- `grep -rn "vmpressure_level" include/lmkd.h` — wrong-header guard.
- `grep -rn "libonnxruntime" Android.bp | grep -v "cc_defaults"` — leak of
  ML dep into unconditional build.
- `grep -rn "predict()" lmkd.cpp | grep -v "use_ml_predictor"` — confirm
  every call is behind the runtime gate.

Run the full Phase 5 A/B once more on the final binary; numbers in
`README_research.md` must match the patch cover-letter to ±2 %.

---

## Cross-Phase Notes

- **Subagent dispatch:** Phases 1, 2, 4 each have a clearly bounded
  fact-gathering task at the top — these are good candidates to dispatch
  via `Explore` before authoring code (e.g. "locate the epoll wiring for
  `mp_event_psi`").
- **Memory:** Phase 0 reference card is the load-bearing artifact. If line
  numbers shift (rebase on upstream), refresh the card before starting any
  later phase.
- **Risk register:** see [plan.md §5](plan.md). Top risk is inference
  latency on low-end hardware — Phase 4 verification gate enforces this.

---

## Phase 6 Deliverables Index

Artifacts produced by the Phase 6 write-up step (single commit on
`feature/ml-psi-predictor`). All paths are repo-root relative.

| Path | One-line description |
|---|---|
| [README_research.md](README_research.md) | Cold-start research README: title/summary citing plan.md success metrics, repo layout, five-step reproduction (build → collect → train → push → bench), `persist.lmk.*` configuration table, text-only PSI→ML→kill architecture diagram, limitations (no-numbers-yet, mem_available approximation, single-device training risk, JSON parser scope), references. |
| [research/upstream/commit-messages.md](research/upstream/commit-messages.md) | Three Gerrit-style commit message drafts for the upstream patch series (scaffolding / injection / runtime toggle), each with `Bug:`, `Test:`, and `Change-Id:` placeholder lines. **Drafts only — no Claude co-author trailer.** |
| [research/upstream/gerrit-howto.md](research/upstream/gerrit-howto.md) | Procedural how-to: remote setup, `commit-msg` hook install, per-commit staging, `git push gerrit HEAD:refs/for/main`, reviewer-add convention, exclude-list for research artifacts. |
| [research/.gitignore](research/.gitignore) | Keeps `data/`, `results/`, `runs/`, `*.onnx`, `*.pt`, `normalization.json`, and Python caches out of git so they never leak into the upstream patch series. |
