# lmkd ML PSI Predictor — Documentation Index

> **Note — read this first.**
> This documentation reflects the **design and target performance envelope**
> for the ML-driven PSI predictor on top of Android's `lmkd`. **No on-device
> benchmark has been run inside this artifact.** Numbers labeled "target"
> come from [plan.md §2](../plan.md); numbers labeled "measured" are
> `<TBD>`. Anywhere you see a chart bar without a value, treat it as an
> empty placeholder, not a result.

## What this branch does

The branch `feature/ml-psi-predictor` augments Android's Low Memory Killer
Daemon (`lmkd`) with an optional LSTM-based PSI (Pressure Stall Information)
predictor. The predictor consumes a rolling 20-sample window of six PSI /
memory features, emits a kill probability, and — when above a configured
threshold — triggers a pre-emptive kill *before* the kernel's static
thresholds would otherwise fire. The entire ML code path is gated behind
`#ifdef LMKD_USE_ML` and the runtime property
`persist.lmk.use_ml_predictor`, so the default build is byte-equivalent to
upstream.

For the long-form, cold-start narrative — research methodology, dataset
design, training pipeline, and Gerrit submission plan — start at
[../README_research.md](../README_research.md). That document is the
authoritative entry point; the files in this `docs/` folder are the
focused technical index built on top of it.

## Table of contents

| # | Document | One-liner |
|---|----------|-----------|
| 01 | [01-changes-summary.md](01-changes-summary.md) | What changed on this branch, file by file, commit by commit. |
| 02 | [02-architecture.md](02-architecture.md) | High-level data flow, ML hook location, `PSIPredictor` class shape. |
| 03 | [03-data-pipeline.md](03-data-pipeline.md) | Sampling, labeling, workloads, dataset shape targets. |
| 04 | [04-model.md](04-model.md) | LSTM architecture, parameter count, training, ONNX export. |
| 05 | [05-integration.md](05-integration.md) | Where the hook lives in `lmkd.cpp`, build flag matrix, fallback rules. |
| 06 | [06-expected-performance.md](06-expected-performance.md) | Target envelopes from plan §2 — **no measured numbers yet.** |
| 07 | [07-rollout.md](07-rollout.md) | Turn it on, turn it off, observe it, debug it. |

## How to read these docs

These docs are written so each audience can read a 2-file slice without
reading the rest:

- **AOSP / lmkd engineer** — skim [02-architecture.md](02-architecture.md)
  for the hook location and class shape, then
  [05-integration.md](05-integration.md) for the build flag matrix and the
  exact file:line anchors in `lmkd.cpp`.
- **ML researcher / data scientist** — skim
  [03-data-pipeline.md](03-data-pipeline.md) for sampling + labeling,
  [04-model.md](04-model.md) for the LSTM and ONNX contract, and
  [06-expected-performance.md](06-expected-performance.md) §6.5 for the
  lead-time hypothesis.
- **Operator / SRE / release engineer** — go straight to
  [07-rollout.md](07-rollout.md) for property reference, observability,
  and failure-mode behavior.

## Honesty footer

Every chart in [06-expected-performance.md](06-expected-performance.md)
that would show measured device data carries a `<TBD>` placeholder. The
A/B bench harness (`research/bench/ab.sh`) exists and is exercised by its
own unit tests, but has not been run end-to-end on a rooted Android device
in this artifact. Do not cite any number from these docs as if it were an
empirical result; cite the plan target instead.
