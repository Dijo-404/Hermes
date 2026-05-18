# research/ — Phase 2 dataset collection

Tooling that samples `/proc/pressure/memory` and `/proc/meminfo` from a
connected Android device, captures lmkd kill events from logcat, and emits
labeled CSVs for training the PSI predictor.

## Layout

| File | Role |
|---|---|
| `collector.py` | Persistent-ADB-shell sampler; writes a raw CSV + a `.kills.log` side-channel. |
| `label.py` | Post-pass labeler: sets `kill_event=1` on rows in the `[T_k - 300ms, T_k - 200ms]` window (defaults). |
| `eda.ipynb` | Sanity-check notebook: class balance, distributions, positive-fraction assertion. |
| `model.py` | Phase 3 — `PSIPredictor` LSTM (6 in → 32 hidden → 1 out, ≤200K params). |
| `dataset.py` | Phase 3 — rolling-window `Dataset`; z-score stats serialised for C++ reuse. |
| `train.py` | Phase 3 — leave-one-scenario-out training + lead-time histogram; emits `psi_predictor.pt`, `normalization.json`, `loso_metrics.csv`, `lead_time_histogram.csv`, filled `model_card.md`. Run with `--quick` for the smoke-test. |
| `export_onnx.py` | Phase 3 — exports the `.pt` checkpoint to ONNX opset 11 with dynamic batch, validates against onnxruntime to within 1e-5. |
| `bench_onnx.py` | Phase 3 — single-sample latency micro-benchmark for the ≤ 2 ms ARM64 p99 budget. |
| `model_card.md` | Template the training script fills in (arch, param count, data sha256, git commit, LOSO metrics, lead histogram). |
| `requirements.txt` | EDA notebook deps + Phase 3 model deps (torch, onnx, onnxruntime, numpy). |
| `notes/` | Phase 1 call-graph and epoll-wiring notes. |
| `data/` | Generated CSVs (gitignored if you choose; not auto-created). |

## Phase 3 — training pipeline (one-paragraph tour)

After Phase 2 has produced one or more labeled CSVs, train with
`python train.py --data data/all.labeled.csv --out-dir out/`. The script
runs leave-one-scenario-out cross-validation (one fold per scenario in the
CSV), trains a final model on all scenarios, asserts the parameter count
is ≤ 200,000, and writes the checkpoint plus normalization sidecar.
Convert to ONNX with `python export_onnx.py --ckpt out/psi_predictor.pt
--out out/psi_predictor.onnx` (asserts ≤ 1e-5 max-diff vs onnxruntime over
100 random windows). Benchmark on the target device with
`python bench_onnx.py --onnx out/psi_predictor.onnx`. Add `--quick` to
`train.py` for the Phase 3 smoke-test (2 epochs, first two scenarios only).

## Install

```
python -m pip install -r requirements.txt   # only needed for the notebook
```

The collector and labeler require only Python 3.11 stdlib + a working
`adb` on PATH.

## Run a collection session

```
adb devices                       # confirm one device, USB debugging on
python collector.py \
    --out      data/web_scroll.csv \
    --duration 600 \
    --device   0A111JECB \
    --scenario web_scroll
```

Outputs:

- `data/web_scroll.csv` — ~10 ms-jitter rows at 100 ms cadence.
- `data/web_scroll.csv.kills.log` — one Unix-epoch float per lmkd kill.

Then label:

```
python label.py \
    --in     data/web_scroll.csv \
    --kills  data/web_scroll.csv.kills.log \
    --out    data/web_scroll.labeled.csv
```

## Workload scenarios (from plan.md §Phase 2)

| Scenario | What to do on the device |
|---|---|
| `idle` | Lockscreen, no apps; baseline. |
| `web_scroll` | Chrome on a long article; thumb-scroll for 10 minutes. |
| `app_switch` | Cycle 5–6 heavy apps via Recents; never let RAM settle. |
| `camera_burst` | Camera app, burst shots + 4K video record. |
| `game_load` | Launch a memory-heavy game (e.g. Genshin), play 10 min. |

## Expected row counts

- 100 ms cadence × duration_seconds × 10 = expected rows.
- A 10-min session yields ~6 000 rows. Five scenarios × ~10 min each
  comfortably clears the Phase 2 verification gate of 50 000 rows.

## CSV schema

```
timestamp_unix, scenario,
some_avg10, some_avg60, some_avg300, some_total,
full_avg10, full_avg60, full_avg300, full_total,
mem_available_kb, swap_free_kb, swap_total_kb,
kill_event
```

`kill_event` is always `0` in the raw collector output; `label.py`
sets it to `1` for rows preceding a kill.

## Verification gate (plan-executable.md Phase 2)

- Combined CSV ≥ 50 000 rows across ≥ 3 scenarios.
- Positive fraction in `[0.005, 0.05]`. The notebook asserts this.
- No missing values. The collector writes empty strings on a parse miss
  and the notebook flags them.
