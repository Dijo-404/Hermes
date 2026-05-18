/*
 * Copyright (C) 2026 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LMKD_ML_PREDICTOR_H_
#define LMKD_ML_PREDICTOR_H_

/*
 * PSIPredictor — ONNX-Runtime backed rolling-window PSI -> kill-probability
 * predictor.  Only compiled when the build is configured with LMKD_USE_ML
 * (see Android.bp).  Lmkd.cpp wraps every reference to this header in
 * `#ifdef LMKD_USE_ML`, so the unmodified build is unaffected.
 *
 * Feature order is locked to match research/dataset.py FEATURES (Phase 3):
 *   [some_avg10, some_avg60, some_total,
 *    full_avg10, full_total, mem_available_kb]
 *
 * Normalization (z-score) statistics are loaded from a sidecar JSON whose
 * format matches NormStats.to_json in research/dataset.py:
 *   {"feature_order": [...], "mean": [...], "std": [...]}
 *
 * ONNX I/O names are fixed by research/export_onnx.py:
 *   input  : "psi_window"   float32 [1, WINDOW, FEATURES]
 *   output : "kill_prob"    float32 [1] (sigmoid-applied)
 */

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

class PSIPredictor {
  public:
    static constexpr int WINDOW = 20;
    static constexpr int FEATURES = 6;
    static constexpr float DEFAULT_KILL_THRESHOLD = 0.65f;

    /*
     * Construct a predictor.  Model + normalization sidecar are loaded
     * lazily on the first push_sample() so init never blocks daemon
     * startup.  If load fails the predictor permanently enters a fatal
     * state and predict() returns -1.0f forever.
     */
    PSIPredictor(const char* model_path, const char* norm_path, float threshold);

    /* Append one raw (un-normalized) feature row.  Z-score normalization
     * is applied here so predict() stays branch-light. */
    void push_sample(float some_avg10, float some_avg60, float some_total,
                     float full_avg10, float full_total, float mem_avail_kb);

    /* Returns sigmoid output in [0,1], or -1.0f if not ready / fatal. */
    float predict();

    /* True iff WINDOW samples have been accumulated since construction
     * and the model loaded successfully. */
    bool ready() const;

    float threshold() const { return threshold_; }

    /* Process-wide singleton accessor.  Returns nullptr if
     * `ro.lmk.use_ml_predictor` was false at startup. */
    static PSIPredictor* instance();

    /* Read system properties and (possibly) construct the singleton.
     * Safe to call multiple times; subsequent calls are no-ops. */
    static void init_from_properties();

  private:
    void ensure_loaded();
    void record_latency_ns(int64_t ns);
    void maybe_log_latency();

    std::string model_path_;
    std::string norm_path_;
    float threshold_;

    /* Lifecycle flags. */
    bool loaded_ = false;
    bool fatal_ = false;

    /* Normalization (per-feature). */
    std::array<float, FEATURES> norm_mean_{};
    std::array<float, FEATURES> norm_std_{};

    /* Ring buffer of normalized samples. */
    std::deque<std::array<float, FEATURES>> ring_;

    /* ONNX Runtime state — initialized lazily inside ensure_loaded(). */
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> mem_info_;

    /* Pre-allocated input tensor backing store (size = WINDOW * FEATURES). */
    std::vector<float> input_buf_;

    /* Latency histogram (simple ring of last N samples in ns). */
    static constexpr size_t LATENCY_RING = 256;
    std::vector<int64_t> latency_ring_;
    size_t latency_idx_ = 0;
    size_t latency_count_ = 0;
    int64_t last_log_mono_ns_ = 0;
};

#endif  // LMKD_ML_PREDICTOR_H_
