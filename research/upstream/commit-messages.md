# Gerrit Commit Message Drafts — `lmkd` ML PSI Predictor Series

> Three-patch upstream series for `platform/system/memory/lmkd` on
> Gerrit (`android-review.googlesource.com`). Drafts only — see
> [gerrit-howto.md](gerrit-howto.md) for upload mechanics. The
> `Change-Id:` line is appended automatically by the Gerrit
> `commit-msg` hook on the first `git commit`; do not hand-write it.
>
> **Do not copy the Claude `Co-Authored-By:` trailer from the
> meta-commit on this fork into these upstream commits** — these are
> authored by the upstream submitter, not by Claude.

---

## Commit A — Scaffolding (no-op when disabled)

```
lmkd: optional ML predictor scaffolding (no-op when disabled)

Adds ml_predictor.{h,cpp} containing PSIPredictor, an ONNX-Runtime
backed rolling-window PSI -> kill-probability classifier, plus a
new `lmkd_ml_defaults` cc_defaults block in Android.bp that is
`enabled: false` by default. The new translation unit and the
`libonnxruntime` dep are pulled in only when a downstream defaults
override flips the block to `enabled: true`, so out-of-the-box
builds are byte-equivalent to a tree that never had this patch.

The predictor is dormant: no lmkd.cpp call sites are added in this
commit. Wiring lands in the follow-up patch so reviewers can audit
the scaffolding (ONNX session lifecycle, normalization sidecar
loader, lazy-load semantics, single-threaded contract) in isolation.

Motivation: lmkd's static PSI thresholds
(ro.lmk.psi_partial_stall_ms, ro.lmk.psi_complete_stall_ms) react
*after* a stall is already underway, costing dropped frames on the
foreground app. A lightweight (~200 K params, p99 < 2 ms target)
predictor can fire 200-500 ms earlier if trained on per-device PSI
traces. This series lays down the inert plumbing; subsequent
patches add the injection point and the runtime toggle.

Bug: <aosp-bug-id>
Test: m lmkd                       # default config, no ML compiled
Test: m lmkd                       # with lmkd_ml_defaults enabled
Test: lmkd_test                    # existing unit suite still green
Change-Id: I<gerrit-hook-generated>
```

---

## Commit B — Injection point inside `__mp_event_psi`

```
lmkd: add PSI prediction injection point in __mp_event_psi

Wires the dormant PSIPredictor from the previous patch into the
PSI event handler. The hook sits at the top of __mp_event_psi,
before the existing static-threshold decision tree, and is guarded
by `#ifdef LMKD_USE_ML` so builds without the ML defaults compile
unchanged. When PSIPredictor::instance() returns nullptr (the
default until the next patch lands the runtime toggle), the entire
branch short-circuits with a single null check.

When active, the hook (a) pushes the current PSI sample
(some_avg10/60/total, full_avg10/total, mem_available_kb) into the
predictor's 20-step ring, and (b) if ready() and predict() >=
threshold(), calls find_and_kill_process() pre-emptively with the
same arguments the static path would have used, then falls through
to the no-kill polling-cadence update. If the model is not ready
or predicts below threshold, control falls through to the existing
static decision tree unchanged - the model augments, it does not
replace.

Motivation: kill latency, not kill count, is what drives
user-perceived jank. Static thresholds are correct on average but
late on the tail; firing earlier on the predicted-stall branch
should reduce dropped frames without materially raising the kill
rate.

Bug: <aosp-bug-id>
Test: lmkd_test
Test: # with predictor disabled (default), behavior diff vs. baseline:
Test: adb logcat -s lmkd:V | diff baseline.log experimental.log
Change-Id: I<gerrit-hook-generated>
```

---

## Commit C — Runtime toggle + bench logging

```
lmkd: add persist.lmk.use_ml_predictor toggle and bench logging

Exposes the ML predictor through four Android system properties so
the same lmkd binary can be A/B compared on the same device without
reflashing. Properties (read by PSIPredictor::init_from_properties()
during lmkd startup):

  persist.lmk.use_ml_predictor   bool, default false
      Master gate. When false, PSIPredictor::instance() returns
      nullptr and the LMKD_USE_ML branch in __mp_event_psi is dead.
  persist.lmk.ml_model_path      string, default
      /system/etc/lmkd/psi_predictor.onnx
  persist.lmk.ml_norm_path       string, default
      /system/etc/lmkd/normalization.json
  persist.lmk.ml_threshold       float, default 0.65
      Sigmoid cutoff for triggering a pre-emptive kill.

Also adds rolling-window p50/p99 latency logging emitted via ALOGI
every 10 s while the predictor is active, so the 2 ms inference
budget can be policed in production.

`persist.` prefix (rather than `ro.`) is deliberate: it lets QA and
field-trial frameworks flip the gate via `adb shell setprop`
between runs without a rebuild.

Motivation: closing the loop on the previous two patches. With the
toggle off, this entire series is dead code on shipping devices.
With it on, OEMs can run the A/B harness in
`research/bench/` (out-of-tree) and ship a per-device threshold.

Bug: <aosp-bug-id>
Test: lmkd_test
Test: adb shell setprop persist.lmk.use_ml_predictor true && \
      adb shell stop lmkd && adb shell start lmkd && \
      adb logcat -s lmkd | grep -E "lmkd-ml|PSIPredictor"
Change-Id: I<gerrit-hook-generated>
```

---

## Notes for the submitter

- The three commits must land in order (A → B → C) because each
  depends on symbols introduced by the previous.
- `Bug:` placeholders must be replaced with a real Buganizer ID
  before `git push gerrit HEAD:refs/for/main`.
- `Change-Id:` is populated automatically by the `commit-msg` hook
  (see [gerrit-howto.md](gerrit-howto.md)). If you `git commit
  --amend`, the same Change-Id is preserved, which is how Gerrit
  groups patch sets on a single review.
- Do **not** include `research/`, `plan.md`, `plan-executable.md`,
  or `README_research.md` in any of these three commits. They live
  in the fork only; the upstream patch series is `ml_predictor.h`,
  `ml_predictor.cpp`, the `Android.bp` defaults block, and the
  `lmkd.cpp` hook plus property reads. See
  [gerrit-howto.md](gerrit-howto.md#what-not-to-push) for the
  exact exclude list.
