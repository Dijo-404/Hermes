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

#define LOG_TAG "lmkd-ml"

#include "ml_predictor.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

#include <cutils/properties.h>
#include <log/log.h>

namespace {

constexpr const char* kPropUseML       = "ro.lmk.use_ml_predictor";
constexpr const char* kPropModelPath   = "ro.lmk.ml_model_path";
constexpr const char* kPropNormPath    = "ro.lmk.ml_norm_path";
constexpr const char* kPropThreshold   = "ro.lmk.ml_threshold";

constexpr const char* kDefaultModel = "/system/etc/lmkd/psi_predictor.onnx";
constexpr const char* kDefaultNorm  = "/system/etc/lmkd/normalization.json";

constexpr const char* kInputName  = "psi_window";
constexpr const char* kOutputName = "kill_prob";

constexpr int64_t kLatencyLogIntervalNs = 10LL * 1000 * 1000 * 1000;  // 10s

PSIPredictor* g_instance = nullptr;
bool          g_init_done = false;

/* ---------------------------------------------------------------------- *
 * Minimal hand-rolled JSON parsing tailored to NormStats sidecar.        *
 *   {"feature_order": [...], "mean": [...], "std": [...]}                *
 * Robust enough for the canonical output of research/dataset.py but      *
 * not a general-purpose parser.                                          *
 * ---------------------------------------------------------------------- */

bool extract_float_array(const std::string& s, const char* key,
                         std::vector<float>* out) {
    out->clear();
    std::string needle = std::string("\"") + key + "\"";
    size_t k = s.find(needle);
    if (k == std::string::npos) return false;
    size_t lb = s.find('[', k);
    if (lb == std::string::npos) return false;
    size_t rb = s.find(']', lb);
    if (rb == std::string::npos) return false;
    std::string body = s.substr(lb + 1, rb - lb - 1);
    /* Split on commas. */
    std::stringstream ss(body);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // trim
        size_t a = 0;
        while (a < tok.size() && std::isspace(static_cast<unsigned char>(tok[a]))) ++a;
        size_t b = tok.size();
        while (b > a && std::isspace(static_cast<unsigned char>(tok[b - 1]))) --b;
        if (a == b) continue;
        out->push_back(std::strtof(tok.substr(a, b - a).c_str(), nullptr));
    }
    return !out->empty();
}

int64_t now_mono_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

}  // namespace

PSIPredictor::PSIPredictor(const char* model_path, const char* norm_path,
                           float threshold)
    : model_path_(model_path ? model_path : ""),
      norm_path_(norm_path ? norm_path : ""),
      threshold_(threshold),
      input_buf_(static_cast<size_t>(WINDOW) * FEATURES, 0.0f),
      latency_ring_(LATENCY_RING, 0) {
    /* Identity normalization until the JSON is loaded. */
    norm_mean_.fill(0.0f);
    norm_std_.fill(1.0f);
}

bool PSIPredictor::ready() const {
    return loaded_ && !fatal_ && ring_.size() == static_cast<size_t>(WINDOW);
}

void PSIPredictor::ensure_loaded() {
    if (loaded_ || fatal_) return;

    /* Load normalization JSON. */
    {
        std::ifstream f(norm_path_);
        if (!f.good()) {
            ALOGE("normalization sidecar unreadable: %s", norm_path_.c_str());
            fatal_ = true;
            return;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        std::string body = ss.str();
        std::vector<float> mean_v, std_v;
        if (!extract_float_array(body, "mean", &mean_v) ||
            !extract_float_array(body, "std",  &std_v)) {
            ALOGE("normalization JSON missing mean/std arrays");
            fatal_ = true;
            return;
        }
        if (mean_v.size() != FEATURES || std_v.size() != FEATURES) {
            ALOGE("normalization arity mismatch: mean=%zu std=%zu expected=%d",
                  mean_v.size(), std_v.size(), FEATURES);
            fatal_ = true;
            return;
        }
        for (int i = 0; i < FEATURES; ++i) {
            norm_mean_[i] = mean_v[i];
            /* Guard divide-by-zero — mirrors NormStats.transform(). */
            norm_std_[i] = (std_v[i] == 0.0f) ? 1.0f : std_v[i];
        }
    }

    /* Init ORT. */
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "lmkd-ml");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), opts);
        mem_info_ = std::make_unique<Ort::MemoryInfo>(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    } catch (const std::exception& e) {
        ALOGE("ONNX model load failed (%s): %s", model_path_.c_str(), e.what());
        env_.reset();
        session_.reset();
        mem_info_.reset();
        fatal_ = true;
        return;
    }

    loaded_ = true;
    last_log_mono_ns_ = now_mono_ns();
    ALOGI("ONNX predictor ready (model=%s, threshold=%.3f)",
          model_path_.c_str(), threshold_);
}

void PSIPredictor::push_sample(float some_avg10, float some_avg60, float some_total,
                               float full_avg10, float full_total, float mem_avail_kb) {
    if (fatal_) return;
    if (!loaded_) {
        ensure_loaded();
        if (fatal_) return;
    }

    std::array<float, FEATURES> raw = {
            some_avg10, some_avg60, some_total,
            full_avg10, full_total, mem_avail_kb,
    };
    std::array<float, FEATURES> normd;
    for (int i = 0; i < FEATURES; ++i) {
        normd[i] = (raw[i] - norm_mean_[i]) / norm_std_[i];
    }

    if (ring_.size() == static_cast<size_t>(WINDOW)) {
        ring_.pop_front();
    }
    ring_.push_back(normd);
}

float PSIPredictor::predict() {
    if (!ready()) return -1.0f;

    /* Pack ring into [1, WINDOW, FEATURES] row-major. */
    size_t idx = 0;
    for (const auto& row : ring_) {
        for (int f = 0; f < FEATURES; ++f) {
            input_buf_[idx++] = row[f];
        }
    }

    const int64_t shape[3] = {1, WINDOW, FEATURES};
    int64_t t0 = now_mono_ns();
    float prob = -1.0f;
    try {
        Ort::Value input = Ort::Value::CreateTensor<float>(
                *mem_info_, input_buf_.data(), input_buf_.size(), shape, 3);
        const char* input_names[]  = {kInputName};
        const char* output_names[] = {kOutputName};
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                     input_names, &input, 1,
                                     output_names, 1);
        if (outputs.empty()) return -1.0f;
        const float* out_data = outputs.front().GetTensorData<float>();
        if (!out_data) return -1.0f;
        prob = out_data[0];
        if (prob < 0.0f) prob = 0.0f;
        if (prob > 1.0f) prob = 1.0f;
    } catch (const std::exception& e) {
        ALOGE("inference failed: %s", e.what());
        return -1.0f;
    }

    record_latency_ns(now_mono_ns() - t0);
    maybe_log_latency();
    return prob;
}

void PSIPredictor::record_latency_ns(int64_t ns) {
    latency_ring_[latency_idx_] = ns;
    latency_idx_ = (latency_idx_ + 1) % LATENCY_RING;
    if (latency_count_ < LATENCY_RING) ++latency_count_;
}

void PSIPredictor::maybe_log_latency() {
    int64_t now = now_mono_ns();
    if (now - last_log_mono_ns_ < kLatencyLogIntervalNs) return;
    last_log_mono_ns_ = now;
    if (latency_count_ == 0) return;

    std::vector<int64_t> tmp(latency_ring_.begin(),
                             latency_ring_.begin() + latency_count_);
    std::sort(tmp.begin(), tmp.end());
    int64_t p50 = tmp[tmp.size() / 2];
    int64_t p99 = tmp[(tmp.size() * 99) / 100];
    ALOGI("inference latency p50=%lld us p99=%lld us (n=%zu)",
          static_cast<long long>(p50 / 1000),
          static_cast<long long>(p99 / 1000),
          latency_count_);
}

PSIPredictor* PSIPredictor::instance() {
    return g_instance;
}

void PSIPredictor::init_from_properties() {
    if (g_init_done) return;
    g_init_done = true;

    char buf[PROPERTY_VALUE_MAX];

    property_get(kPropUseML, buf, "false");
    bool use_ml = (strcmp(buf, "true") == 0 || strcmp(buf, "1") == 0);
    if (!use_ml) {
        ALOGI("ML predictor disabled via %s", kPropUseML);
        return;
    }

    char model_path[PROPERTY_VALUE_MAX];
    char norm_path[PROPERTY_VALUE_MAX];
    property_get(kPropModelPath, model_path, kDefaultModel);
    property_get(kPropNormPath,  norm_path,  kDefaultNorm);

    property_get(kPropThreshold, buf, "");
    float threshold = DEFAULT_KILL_THRESHOLD;
    if (buf[0] != '\0') {
        float t = std::strtof(buf, nullptr);
        if (t > 0.0f && t < 1.0f) threshold = t;
    }

    g_instance = new PSIPredictor(model_path, norm_path, threshold);
    ALOGI("ML predictor configured (model=%s norm=%s threshold=%.3f) — model "
          "will be loaded lazily on first sample",
          model_path, norm_path, threshold);
}
