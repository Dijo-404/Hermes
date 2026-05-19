# 02 — Architecture

## High-level data flow

```mermaid
flowchart TD
    K["kernel<br/>/proc/pressure/memory"] -- "EPOLLPRI" --> EP["epoll_wait<br/>(main loop)"]
    EP --> CH["call_handler()<br/>lmkd.cpp:3965"]
    CH --> MP["mp_event_psi()<br/>lmkd.cpp:3173"]
    MP --> MPI["__mp_event_psi()<br/>lmkd.cpp:2717"]
    MPI --> HOOK{"#ifdef LMKD_USE_ML<br/>ML hook<br/>lmkd.cpp:2936"}
    HOOK -- "predictor fires<br/>(prob >= threshold)" --> KILL["find_and_kill_process()<br/>lmkd.cpp:2539"]
    HOOK -- "fall through<br/>(not ready / below threshold /<br/>predictor null)" --> STATIC["static threshold checks<br/>thrashing + watermark"]
    STATIC -- "static path decides<br/>to kill" --> KILL
    KILL --> REAPER["reaper thread<br/>(SIGKILL + process_mrelease)"]
    STATIC -- "no kill needed" --> POLL["update polling cadence"]
```

The ML hook is positioned **inside** `__mp_event_psi` so it sees the same
parsed PSI sample (and the same `mi` / `wi` snapshots) that the static
decision tree would use one block later. Pre-emptive kills go through the
exact same `find_and_kill_process` victim-selection path — only the
*trigger* changes.

## ML hook decision tree

```mermaid
flowchart TD
    A["__mp_event_psi enters ML hook<br/>lmkd.cpp:2936"] --> B{"PSIPredictor::instance() != nullptr?"}
    B -- "no<br/>(persist.lmk.use_ml_predictor=false<br/>or LMKD_USE_ML undef)" --> X1["fall through to static path"]
    B -- "yes" --> C["push_sample(6 features)"]
    C --> D{"ready()?<br/>(>= 20 samples)"}
    D -- "no" --> X2["fall through"]
    D -- "yes" --> E["predict() -> prob"]
    E --> F{"prob >= threshold?<br/>(default 0.65)"}
    F -- "no" --> X3["fall through"]
    F -- "yes" --> G["log 'pre-emptive kill (p=…)'<br/>call find_and_kill_process<br/>goto no_kill"]
```

The four fall-through conditions are listed exhaustively in
[05-integration.md](05-integration.md#fallback-behavior).

## PSIPredictor class shape

```mermaid
classDiagram
    class PSIPredictor {
      +static constexpr int WINDOW = 20
      +static constexpr int FEATURES = 6
      +static constexpr float DEFAULT_KILL_THRESHOLD = 0.65f
      +PSIPredictor(model_path, norm_path, threshold)
      +push_sample(some_avg10, some_avg60, some_total, full_avg10, full_total, mem_avail_kb) noexcept
      +predict() float noexcept
      +ready() bool noexcept
      +threshold() float noexcept
      +static instance() PSIPredictor*
      +static init_from_properties() void
      -ensure_loaded() void
      -record_latency_ns(int64_t) void
      -maybe_log_latency() void
      -string model_path_
      -string norm_path_
      -float threshold_
      -bool loaded_
      -bool fatal_
      -array~float,6~ norm_mean_
      -array~float,6~ norm_std_
      -deque~array~float,6~~ ring_
      -unique_ptr~Ort::Env~ env_
      -unique_ptr~Ort::Session~ session_
      -unique_ptr~Ort::MemoryInfo~ mem_info_
      -vector~float~ input_buf_
      -vector~int64_t~ latency_ring_
      -size_t latency_idx_
      -size_t latency_count_
      -int64_t last_log_mono_ns_
    }
```

Class declaration: [`ml_predictor.h:62`](../ml_predictor.h#L62).

Key contract notes:

- All four mutator/observer methods (`push_sample`, `predict`, `ready`,
  `threshold`) are marked `noexcept`. Any exception escaping ORT is caught
  internally; `predict()` returns `-1.0f` on failure and the caller treats
  that the same as "not ready" — i.e. fall through to the static path.
- The singleton is owned by a function-local `std::unique_ptr` inside
  `instance()`. It is constructed at most once via `init_from_properties()`
  (call-once semantics) and destroyed at process exit, which releases the
  Ort::Env and Ort::Session in deterministic order.
- The model and normalization stats are loaded **lazily** on the first
  `push_sample()` so daemon startup is never blocked by I/O. If load
  fails, `fatal_` is latched and `instance()` continues to return a
  pointer whose `ready()` is permanently `false`.
- Threading: single-owner contract. `lmkd`'s main loop is the sole caller;
  `instance()` itself is thread-safe (call_once), but the returned object
  is not internally synchronized.

## What was not changed

The branch deliberately leaves the following files untouched. Anyone
auditing the diff can confirm with `git diff main..HEAD -- <path>`:

- [`include/lmkd.h`](../include/lmkd.h) — public protocol header.
- [`statslog.h`](../statslog.h), [`statslog.cpp`](../statslog.cpp) —
  kill-stat reporting (the ML pre-emptive kill funnels through the same
  `find_and_kill_process` and so inherits existing statslog behavior).
- [`reaper.h`](../reaper.h), [`reaper.cpp`](../reaper.cpp) — async
  SIGKILL + `process_mrelease` worker.
- [`watchdog.h`](../watchdog.h), [`watchdog.cpp`](../watchdog.cpp) —
  daemon watchdog.
- [`libpsi/`](../libpsi) — the PSI fd wiring layer
  (`init_psi_monitor`, `register_psi_monitor`).
- [`liblmkd_utils.cpp`](../liblmkd_utils.cpp) — client-side helpers.
- [`lmkd.rc`](../lmkd.rc) — init script.
- [`OWNERS`](../OWNERS), [`PREUPLOAD.cfg`](../PREUPLOAD.cfg) — Gerrit
  metadata.
- [`event.logtags`](../event.logtags) — atom IDs.

The absence of changes to `event.logtags` and `statslog.{h,cpp}` is
intentional: the ML pre-emptive kill emits the *same* logtag with
`kill_desc = "ml predictor pre-emptive kill"`, so downstream analytics
pipelines (`westworld`, `clearcut`) do not need a new schema.
