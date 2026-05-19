# 07 — Rollout

This document is for the operator turning the ML predictor on, observing
it, and turning it back off if anything goes wrong.

## Turning it on

1. **Build with the ML cc_defaults enabled.** In
   [`Android.bp`](../Android.bp), flip `enabled: true` on
   `lmkd_ml_defaults` (or build with a vendor overlay that does), then
   from the AOSP tree:

   ```
   $ source build/envsetup.sh
   $ lunch aosp_<device>-userdebug
   $ m lmkd
   ```

   The resulting binary will define `LMKD_USE_ML` and link against
   `libonnxruntime`.

2. **Push the model and normalization sidecar.**

   ```
   $ adb root && adb remount
   $ adb push psi_predictor.onnx       /system/etc/lmkd/psi_predictor.onnx
   $ adb push normalization.json       /system/etc/lmkd/normalization.json
   ```

   Default paths are baked into [`ml_predictor.cpp`](../ml_predictor.cpp);
   non-default paths can be overridden by setting
   `persist.lmk.ml_model_path` and `persist.lmk.ml_norm_path`.

3. **Enable the runtime flag.**

   ```
   $ adb shell setprop persist.lmk.use_ml_predictor true
   ```

4. **Reboot.** `init_from_properties()` runs once at lmkd startup, so the
   property must be set *before* the daemon reads it.

   ```
   $ adb reboot
   ```

After reboot, confirm with `logcat -s lmkd-ml:I` (see
[Observability](#observability)).

## Property reference

| Property | Type | Default | Purpose |
|----------|------|---------|---------|
| `persist.lmk.use_ml_predictor` | bool | `false` | Master switch. If `false`, `instance()` returns `nullptr` and the daemon behaves as upstream. |
| `persist.lmk.ml_model_path` | string | `/system/etc/lmkd/psi_predictor.onnx` | Filesystem path to the exported ONNX graph. |
| `persist.lmk.ml_norm_path` | string | `/system/etc/lmkd/normalization.json` | Filesystem path to the z-score normalization sidecar. |
| `persist.lmk.ml_threshold` | float (parsed string) | `0.65` | Decision threshold; `predict()` must return `≥` this for a pre-emptive kill. |

The four properties are read once by `init_from_properties()`; changing
them at runtime has no effect until the daemon is restarted. Source:
[`ml_predictor.cpp:38-41`](../ml_predictor.cpp#L38).

## Turning it off

```
$ adb shell setprop persist.lmk.use_ml_predictor false
$ adb reboot
```

After reboot, `init_from_properties()` reads the property, sees `false`,
and never constructs the singleton. Every site in the hook that does
`if (PSIPredictor* ml = PSIPredictor::instance())` immediately
short-circuits, and the daemon falls through to the unchanged static
threshold path. No code is uninstalled, no library is unloaded — the ML
path is simply quiescent.

If you need to physically prevent the ML path from being even *possible*,
rebuild lmkd without `LMKD_USE_ML` defined; the resulting binary is
byte-equivalent to upstream.

## Observability

The predictor logs to the standard Android logger under the
`lmkd-ml` tag. Useful filters:

```
$ adb logcat -s lmkd-ml:I
```

Lines you should expect to see, in order, after a successful boot:

1. `lmkd-ml: init: predictor enabled (model=…, norm=…, threshold=0.65)`
   — emitted at the bottom of `init_from_properties()`.
2. `lmkd-ml: model loaded (params=5153, opset=11)` — emitted by
   `ensure_loaded()` the first time `push_sample()` finishes loading.
3. `lmkd-ml: pre-emptive kill triggered (p=0.872)` — emitted at
   [`lmkd.cpp:2966`](../lmkd.cpp#L2966) on every pre-emptive fire.
4. `lmkd-ml: inf p50=0.41ms p99=1.18ms (n=256)` — emitted by
   `maybe_log_latency()` every 10 s once enough samples have accumulated.

Quick health check (no model loaded, no kills triggered → fall-through path):

```
$ adb shell setprop persist.lmk.use_ml_predictor true
$ adb shell stop lmkd && adb shell start lmkd
$ adb logcat -s lmkd-ml:I -d
```

If you see line 1 but never see line 2 within a couple of minutes of
moderate device usage, jump to [Failure-mode reference](#failure-mode-reference).

## Failure-mode reference

Every failure mode degrades the daemon to **upstream-equivalent
behavior**. There is no path here that *stops* lmkd from killing — only
paths that stop the ML augmentation from helping.

| Symptom | Cause | Effect |
|---------|-------|--------|
| No `model loaded` log line | `persist.lmk.ml_model_path` missing or pointing at a nonexistent file. | `ensure_loaded()` sets `fatal_=true`; `predict()` returns `-1.0f` forever; hook always falls through. |
| `lmkd-ml: norm load failed` in logcat | `normalization.json` missing, unreadable, or malformed (schema mismatch). | Same as above: `fatal_=true`, static path runs. |
| `lmkd-ml: session init threw` in logcat | ONNX Runtime threw during `Ort::Session` construction (opset mismatch, corrupt graph, OOM). | Caught inside `ensure_loaded()`; `fatal_=true`, static path runs. |
| Latency log lines vanish for >30 s | `predict()` throwing repeatedly. | Each throw is caught inside `predict()`, returns `-1.0f`, counted as not-ready. Static path runs every cycle. |
| Kill rate jumps above pre-rollout baseline | False-positive rate too high (Precision < 0.70). | Raise `persist.lmk.ml_threshold` from 0.65 → 0.75; reboot; re-evaluate. |
| RSS delta exceeds 4 MB | ORT arena over-allocating. | Set `persist.lmk.use_ml_predictor=false`; reboot; file an issue with the breakdown table from [06-expected-performance.md §6.4](06-expected-performance.md#64-memory-overhead-budget). |

When in doubt, the safest revert is one line:
`adb shell setprop persist.lmk.use_ml_predictor false && adb reboot`.
