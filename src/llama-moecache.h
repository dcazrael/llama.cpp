#pragma once

#include <cstdint>

struct llama_model;
struct ggml_tensor;
struct moe_cache; // opaque, defined in llama-moecache.cpp

struct llama_moe_cache_layer {
    int il = -1;

    int32_t n_slots = 0;

    // host-resident source weights (the authoritative experts)
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    // device-resident cache slots, ne[2] == n_slots + 1 (last slot all zeros)
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // expert id -> slot (or n_slots when uncached); I32 [1, n_expert]
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;
};

// aggregate cache telemetry across all layers; safe to call on a nullptr cache
struct llama_moe_cache_stats {
    uint64_t steps             = 0; // total decode steps observed
    uint64_t hits              = 0; // expert routed to a device-resident slot
    uint64_t misses            = 0; // expert routed while uncached
    uint64_t uploads           = 0; // new device uploads completed
    uint64_t evictions         = 0; // resident experts evicted to make room
    uint64_t bytes_uploaded    = 0; // host->device bytes moved by uploads
    uint64_t bytes_served      = 0; // device-resident expert bytes used by hits
    uint64_t gated             = 0; // uncached observations held back by admission gate
};

// create the cache for every host-resident expert layer of the model.
// admit_threshold: recent-use sightings required before an uncached expert
//                  is queued for upload (1 = unconditional admission).
// returns 0 on success, non-zero on failure.
// safe to call more than once with the same/out pointer (noop after success).
int llama_moe_cache_create(moe_cache ** out, const llama_model & model, int32_t n_slots, int32_t max_inserts, int32_t admit_threshold);

// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer * llama_moe_cache_lookup(const moe_cache * mc, const ggml_tensor * up_exps);

// apply throttled LRU updates; call after synchronize, between graph executions only
void llama_moe_cache_step(moe_cache * mc);

// tear down the cache and stop the upload worker thread; safe to call more than once
void llama_moe_cache_destroy(moe_cache * mc);

// snapshot aggregate cache telemetry; safe to call on a nullptr cache
void llama_moe_cache_get_stats(const moe_cache * mc, llama_moe_cache_stats * out);
