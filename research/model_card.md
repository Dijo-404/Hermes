# psi_predictor — model card

> Auto-filled by `research/train.py`. Placeholders `{{...}}` are replaced
> at the end of a successful training run; if you see literal `{{...}}`
> in your copy, training did not complete and the file is the raw template.

## Purpose

Predicts the probability of an imminent lmkd kill in the next ~200 ms
from a 20-sample sliding window of PSI + meminfo features. Output drives
the optional ML branch in `__mp_event_psi` (see Phase 4); the static
threshold path is preserved as fallback.

## Architecture

```
Input  [B, 20, 6]
  features = (some_avg10, some_avg60, some_total,
              full_avg10, full_total, mem_available_kb)
   |
   v
LSTM(input=6, hidden=32, num_layers=1, batch_first=True)
   |
   v   take last timestep -> [B, 32]
Dropout(p=0.2)
   |
   v
Linear(32 -> 1)
   |
   v
Sigmoid       -> [B, 1]   kill_prob ∈ [0, 1]
```

## Parameter count

`{{PARAM_COUNT}}` parameters (budget: ≤ 200,000).

Closed-form breakdown:

- LSTM: `4 * (input*hidden + hidden*hidden + 2*hidden)`
        = `4 * (6*32 + 32*32 + 2*32)` = **5,120**
- Linear: `hidden + 1` = **33**
- Total: **5,153**

## Training data

- CSV path hashed at training time.
- SHA-256: `{{DATA_SHA256}}`
- Source: `research/collector.py` + `research/label.py`
  (label window: `[T_k - 300 ms, T_k - 200 ms]` by default).

## Training code

- Git commit: `{{GIT_COMMIT}}`
- Loss: `BCEWithLogitsLoss(pos_weight=10.0)` (class-weighted; matches
  plan-executable.md Phase 3 "positive weight ≈ 10").
- Optimizer: Adam, lr=1e-3.
- Normalization: per-feature z-score, statistics fit on train split,
  serialised to `normalization.json` for C++ reuse.

## LOSO cross-validation results

| Held-out scenario | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|
{{LOSO_FOLDS_TABLE}}

- Mean precision (LOSO): `{{LOSO_MEAN_PRECISION}}`  (target ≥ 0.70)
- Mean recall    (LOSO): `{{LOSO_MEAN_RECALL}}`    (target ≥ 0.85)

## Lead-time

- Histogram CSV: `{{LEAD_HISTOGRAM_PATH}}`
- Fraction of true positives firing ≥ 100 ms before the kill:
  `{{LEAD_GE_100MS_FRAC}}` (target ≥ 0.80).

## ONNX export

- Opset: `{{ONNX_OPSET}}` (Android ONNX Runtime prebuilts safe).
- Input name: `psi_window`, shape `[batch, 20, 6]`, batch axis dynamic.
- Output name: `kill_prob`, shape `[batch, 1]`.
- Validation: max-abs-diff PyTorch vs onnxruntime over 100 random windows
  must be < 1e-5 (`research/export_onnx.py` asserts).

## Intended use & limitations

- Augments, does not replace, the static-threshold path in lmkd. When the
  predicted probability is below `ro.lmk.ml_threshold` (default 0.65),
  control falls through to the existing PSI threshold logic.
- Trained on a small set of synthetic / on-device workloads. Distribution
  shift across device classes is the largest known risk (plan.md §5,
  Risk Register row 4) — re-evaluate per new device class.
- Inference budget: p99 ≤ 2 ms on Pixel 4a CPU (ARM64). Verified by
  `research/bench_onnx.py` on target hardware.
