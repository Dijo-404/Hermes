# plan.md — ML-Driven Predictive PSI Tuning for lmkd

**Project:** Replacing static PSI threshold heuristics in Android's Low Memory Killer Daemon with a lightweight on-device LSTM inference layer  
**Base Repo:** `https://android.googlesource.com/platform/system/memory/lmkd`  
**Branch Target:** `android-latest-release`  
**Author:** Dijo S Benelen  
**Duration:** ~10 weeks

---

## 1. Problem Statement

The current `lmkd` daemon reacts to memory pressure using static PSI thresholds hardcoded in hardware configuration files (e.g., `ro.lmk.psi_partial_stall_ms = 70ms` for high-end devices). This one-size-fits-all approach fails in two directions:

- **Too low** → unnecessary background process kills, degraded multitasking
- **Too high** → system lags before pressure relief arrives, user-visible jank

The root cause is that PSI threshold tuning is a reactive system with no awareness of *upcoming* memory allocation patterns. The goal of this project is to replace the static threshold with a lightweight LSTM model that predicts thrashing events 200–500ms before they occur, allowing lmkd to take pre-emptive action.

---

## 2. Research Hypothesis

> A lightweight LSTM model trained on PSI time-series, `oom_adj` score distributions, and foreground app memory allocation deltas can predict memory pressure breaches with sufficient lead time to eliminate user-perceived lag, without increasing background process kill frequency.

**Success metrics:**
- ≥ 30% reduction in UI jank events (dropped frames measured via `dumpsys gfxinfo`)
- ≤ 5% increase in background process kill frequency vs. static baseline
- Model inference latency ≤ 2ms on-device (mandatory — lmkd is latency-critical)
- RSS overhead of inference engine ≤ 4MB

---

## 3. Codebase Map

```
platform/system/memory/lmkd/
├── lmkd.cpp              ← PRIMARY — PSI event handler, kill decision logic
├── reaper.cpp            ← Process termination execution
├── statslog.cpp          ← Kill event telemetry (used for dataset labeling)
├── watchdog.cpp          ← Daemon health monitoring
├── libpsi/
│   └── psi.cpp           ← PSI monitor subscription logic (READ THIS FIRST)
├── include/
│   └── lmkd.h            ← oom_adj structs, pressure level enums
└── Android.bp            ← Build config (where ONNX Runtime dep gets added)
```

**Key functions to understand before writing any code:**

| Function | File | Purpose |
|---|---|---|
| `mp_event_psi()` | `lmkd.cpp` | Fires on PSI threshold breach — **injection point** |
| `find_and_kill_process()` | `lmkd.cpp` | Kill decision logic to defer/replace |
| `init_psi_monitor()` | `libpsi/psi.cpp` | Sets up kernel PSI subscriptions |
| `poll_kernel()` | `lmkd.cpp` | Main epoll loop — understand threading model |

---

## 4. Phases

---

### Phase 1 — Environment Setup & Codebase Familiarization
**Duration:** Week 1  
**Goal:** Get lmkd building and running in isolation; understand the full PSI event lifecycle.

#### Tasks

- [ ] Clone the canonical repo
  ```bash
  git clone https://android.googlesource.com/platform/system/memory/lmkd
  cd lmkd
  git checkout android-latest-release
  ```

- [ ] Set up AOSP build environment (Ubuntu 22.04 recommended)
  ```bash
  sudo apt install repo git-core gnupg flex bison build-essential \
    zip curl zlib1g-dev libc6-dev-i386 libncurses5 lib32z1 \
    libgl1-mesa-dev libxml2-utils xsltproc unzip
  ```

- [ ] Read through `lmkd.cpp` top to bottom — annotate every PSI-related code path

- [ ] Enable lmkd debug logs on a rooted Android device or emulator:
  ```bash
  adb shell setprop ro.lmk.debug true
  adb logcat -s lmkd
  ```

- [ ] Manually stress-test PSI thresholds:
  ```bash
  # Lower threshold to force frequent PSI events
  adb shell setprop ro.lmk.psi_partial_stall_ms 10
  # Run memory pressure workload
  adb shell stress-ng --vm 2 --vm-bytes 80% --timeout 60s
  ```

- [ ] Read the PSI kernel documentation:  
  `https://www.kernel.org/doc/html/latest/accounting/psi.html`

**Deliverable:** Written architecture notes on how `mp_event_psi()` → `find_and_kill_process()` flows, with annotated call graph.

---

### Phase 2 — Dataset Collection Pipeline
**Duration:** Week 2–3  
**Goal:** Collect labeled time-series data linking PSI metrics to kill events.

#### Data sources

| Signal | Source | Sampling Rate |
|---|---|---|
| `some` / `full` PSI stall (ms) | `/proc/pressure/memory` | 100ms |
| ZRAM usage | `/proc/meminfo` (`SwapTotal`, `SwapFree`) | 100ms |
| Foreground app RSS | `/proc/<pid>/status` | 100ms |
| `oom_adj` score distribution | `/proc/<pid>/oom_score_adj` | 500ms |
| Kill events (label) | `statslog.cpp` output / `logcat` | event-driven |

#### Collection script (Python, runs on host via ADB)

```python
# collector.py — run during stress workload sessions
import subprocess, time, csv, datetime

def read_psi():
    out = subprocess.check_output(
        ['adb', 'shell', 'cat', '/proc/pressure/memory']
    ).decode()
    # Parse: "some avg10=X.XX avg60=X.XX avg300=X.XX total=XXXXXX"
    lines = out.strip().split('\n')
    return {l.split()[0]: dict(p.split('=') for p in l.split()[1:]) for l in lines}

def read_meminfo():
    out = subprocess.check_output(
        ['adb', 'shell', 'cat', '/proc/meminfo']
    ).decode()
    return {l.split(':')[0]: l.split(':')[1].strip() for l in out.strip().split('\n')}

with open('psi_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'timestamp', 'some_avg10', 'some_avg60', 'some_total',
        'full_avg10', 'full_total', 'memavail_kb', 'swap_free_kb', 'kill_event'
    ])
    while True:
        psi = read_psi()
        mem = read_meminfo()
        writer.writerow([
            datetime.datetime.now().isoformat(),
            psi['some']['avg10'], psi['some']['avg60'], psi['some']['total'],
            psi['full']['avg10'], psi['full']['total'],
            mem.get('MemAvailable', '0').replace(' kB',''),
            mem.get('SwapFree', '0').replace(' kB',''),
            0  # label — post-process with kill timestamps from logcat
        ])
        time.sleep(0.1)
```

#### Workload scenarios to record

- Idle baseline (10 min)
- Heavy browser tab switching (30 tabs, Chrome)
- Gaming session (high RAM game, background music app)
- Camera + video processing concurrent
- Low-RAM device simulation (`adb shell setprop ro.config.low_ram true`)

**Target dataset size:** ≥ 50,000 timesteps across varied workloads.  
**Label creation:** Post-process CSV — mark `kill_event=1` at T-200ms before each kill observed in logcat.

**Deliverable:** `psi_dataset.csv` with labeled kill events, basic EDA notebook.

---

### Phase 3 — Model Design and Training
**Duration:** Week 3–4  
**Goal:** Train a lightweight LSTM that predicts imminent kill necessity ≥ 200ms ahead.

#### Architecture

```
Input window: 20 timesteps × 6 features (2s of 100ms samples)
    → LSTM(hidden=32, layers=1)
    → Dropout(0.2)
    → Linear(32 → 1)
    → Sigmoid
Output: P(kill_needed_in_next_200ms)
```

Kept deliberately small — target is ≤ 200K parameters to meet the 2ms inference and 4MB RSS constraints.

#### Training (PyTorch)

```python
import torch
import torch.nn as nn

class PSIPredictor(nn.Module):
    def __init__(self, input_size=6, hidden=32, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return torch.sigmoid(self.fc(out))

# Training loop with class-weighted BCE loss
# (kill events are rare — weight positive class ~10x)
criterion = nn.BCELoss(weight=torch.tensor([10.0]))
```

#### Export to ONNX (for C++ inference in lmkd)

```python
dummy = torch.randn(1, 20, 6)
torch.onnx.export(
    model, dummy, "psi_predictor.onnx",
    input_names=["psi_window"],
    output_names=["kill_prob"],
    opset_version=11
)
```

#### Evaluation metrics

- Precision / Recall on kill prediction (target: Recall ≥ 0.85, Precision ≥ 0.70)
- Lead time distribution — how many ms before actual kill event does model fire?
- False positive rate — how often does model predict kill when not needed?

**Deliverable:** `psi_predictor.onnx`, training notebook, evaluation report.

---

### Phase 4 — C++ Inference Integration into lmkd
**Duration:** Week 5–6  
**Goal:** Embed ONNX Runtime into lmkd; replace static PSI threshold with model output.

#### Dependency addition (`Android.bp`)

```bp
cc_binary {
    name: "lmkd",
    // existing entries...
    shared_libs: [
        "libonnxruntime",   // add this
        "liblog",
        "libcutils",
    ],
    static_libs: [
        "libpsi",
    ],
}
```

#### New file: `ml_predictor.h`

```cpp
#pragma once
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <deque>
#include <array>

class PSIPredictor {
public:
    static constexpr int WINDOW = 20;
    static constexpr int FEATURES = 6;
    static constexpr float KILL_THRESHOLD = 0.65f;

    PSIPredictor(const char* model_path);
    float predict();
    void push_sample(float some_avg10, float some_avg60, float some_total,
                     float full_avg10, float full_total, float mem_avail_kb);

private:
    Ort::Session session_;
    std::deque<std::array<float, FEATURES>> window_;
    Ort::Env env_;
};
```

#### Injection point in `lmkd.cpp`

The `mp_event_psi()` function currently fires when kernel PSI crosses the static threshold. Modify it to consult the model *before* deciding to kill:

```cpp
// BEFORE (static threshold only):
static void mp_event_psi(int data, uint32_t events, struct polling_params *poll_params) {
    // ... reads PSI, decides to kill based on static oom_adj thresholds
    find_and_kill_process(params, ...);
}

// AFTER (ML-augmented):
static void mp_event_psi(int data, uint32_t events, struct polling_params *poll_params) {
    // Push current PSI sample to model's rolling window
    g_predictor->push_sample(
        psi_data.some.avg10, psi_data.some.avg60, psi_data.some.total,
        psi_data.full.avg10, psi_data.full.total, mi.field.nr_free_pages
    );

    float kill_prob = g_predictor->predict();

    if (kill_prob >= PSIPredictor::KILL_THRESHOLD) {
        // Pre-emptive kill — model predicts pressure before PSI breach
        ALOGI("lmkd-ml: pre-emptive kill triggered (p=%.3f)", kill_prob);
        find_and_kill_process(params, ...);
    } else if (/* static PSI threshold still breached */) {
        // Fallback: static path still active as safety net
        find_and_kill_process(params, ...);
    }
    // else: model predicts safe — skip kill this cycle
}
```

**Critical constraint:** Inference must complete within the epoll timeout window. Profile inference time aggressively — if > 2ms on target hardware, reduce LSTM hidden size or quantize to INT8.

**Deliverable:** Modified `lmkd.cpp`, `ml_predictor.h`, `ml_predictor.cpp`, updated `Android.bp`.

---

### Phase 5 — Testing and Benchmarking
**Duration:** Week 7–8  
**Goal:** Quantify improvement over static baseline on real hardware.

#### Test device requirements

- Rooted Android device (Pixel 4a or equivalent — affordable, well-documented)
- OR Android emulator with memory constraints configured
- AOSP build environment for flashing modified lmkd binary

#### Benchmark suite

| Test | Tool | Metric |
|---|---|---|
| UI jank under memory pressure | `dumpsys gfxinfo <pkg>` | Janky frames % |
| Kill frequency comparison | Modified `statslog.cpp` | Kills/hour |
| Inference latency | `clock_gettime()` bracketing predict() call | p50/p99 ms |
| Memory overhead | `/proc/lmkd/status` RSS delta | KB |
| Cold start latency (regression check) | `adb shell am start -W` | totalTime ms |

#### A/B test setup

```bash
# Baseline: static thresholds
adb shell setprop ro.lmk.psi_partial_stall_ms 70
adb shell setprop ro.lmk.use_ml_predictor false

# Experimental: ML predictor enabled
adb shell setprop ro.lmk.use_ml_predictor true
```

Add a `ro.lmk.use_ml_predictor` property as a runtime toggle so baseline and experimental can be compared on the same device without reflashing.

**Deliverable:** Benchmark results table, `results/` directory with raw logs, analysis notebook.

---

### Phase 6 — Write-Up and Contribution Prep
**Duration:** Week 9–10  
**Goal:** Package results as a publishable research artifact and upstream-ready patch series.

#### Research paper outline

1. Abstract
2. Background — lmkd architecture, PSI mechanics, limitations of static tuning
3. Methodology — dataset collection, LSTM architecture, integration design
4. Evaluation — benchmark results vs. baseline
5. Discussion — failure modes, generalizability across OEM hardware
6. Related Work — prior ML scheduling work (cite EAS Bayesian optimization, WALT)
7. Conclusion + Future Work (ZRAM deduplication + UFFD synergy as next step)

#### Upstream patch format

AOSP contributions go via Gerrit, not GitHub PRs:
```bash
# One-time setup
git clone https://android.googlesource.com/platform/system/memory/lmkd
git remote add gerrit https://android.googlesource.com/platform/system/memory/lmkd

# Commit with Change-Id (required by Gerrit)
git commit -m "lmkd: add ML-driven predictive PSI threshold tuning

Replace static psi_partial_stall_ms with an LSTM inference layer
that predicts memory thrashing 200ms ahead, reducing UI jank by
X% while maintaining kill frequency parity with static baseline.

Bug: <aosp-bug-id>
Test: lmkd_test, manual benchmark on Pixel 4a (Android 15)"

# Push to Gerrit for review
git push gerrit HEAD:refs/for/main
```

**Deliverable:** Draft paper (LaTeX), clean patch series, `README_research.md` documenting reproduction steps.

---

## 5. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Inference latency > 2ms on low-end hardware | Medium | Quantize ONNX model to INT8; reduce hidden size to 16 if needed |
| Insufficient kill events in dataset for training | Medium | Synthesize pressure events with `stress-ng`; augment with ZRAM exhaustion workloads |
| ONNX Runtime adds too much binary size for AOSP | Low | Use TensorFlow Lite runtime instead — smaller footprint, already in AOSP tree |
| Model overfits to specific device memory profile | Medium | Train on data from ≥ 3 different RAM configurations (3GB, 6GB, 8GB) |
| Rooted test device unavailable | Low | Use Android emulator with `-memory` flag; PSI is readable in emulator |

---

## 6. Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| ONNX Runtime (Android) | ≥ 1.17 | C++ inference engine in lmkd |
| PyTorch | ≥ 2.1 | Model training |
| Python | 3.11 | Dataset collection + training scripts |
| Android NDK | r26+ | Cross-compile for ARM64 |
| ADB + rooted device | — | Live PSI data collection and testing |
| `stress-ng` | latest | Synthetic memory pressure workloads |

---

## 7. Directory Layout (Your Fork)

```
android_system_memory_lmkd/
├── lmkd.cpp                  ← modified
├── ml_predictor.h            ← new
├── ml_predictor.cpp          ← new
├── Android.bp                ← modified
├── research/
│   ├── collector.py          ← PSI data collection script
│   ├── train.py              ← LSTM training
│   ├── evaluate.py           ← benchmark analysis
│   ├── psi_dataset.csv       ← collected data (gitignored if large)
│   ├── psi_predictor.onnx    ← trained model artifact
│   └── results/              ← benchmark outputs
├── README_research.md        ← reproduction guide
└── plan.md                   ← this file
```

---

## 8. Key References

- AOSP lmkd source: `https://android.googlesource.com/platform/system/memory/lmkd`
- PSI kernel docs: `https://www.kernel.org/doc/html/latest/accounting/psi.html`
- AOSP lmkd documentation: `https://source.android.com/docs/core/perf/lmkd`
- Memory Management on Mobile (ISMM 2024): `https://www.steveblackburn.org/pubs/papers/android-ismm-2024-errata.pdf`
- ONNX Runtime Android: `https://onnxruntime.ai/docs/build/android.html`
- NanoTag (MTE reference, adjacent work): `https://arxiv.org/html/2509.22027v3`
