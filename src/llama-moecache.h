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

// create the cache for every host-resident expert layer of the model.
// returns 0 on success, non-zero on failure.
// safe to call more than once with the same/out pointer (noop after success).
int llama_moe_cache_create(moe_cache ** out, const llama_model & model, int32_t n_slots, int32_t max_inserts);

// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer * llama_moe_cache_lookup(const moe_cache * mc, const ggml_tensor * up_exps);

// apply throttled LRU updates; call after synchronize, between graph executions only
void llama_moe_cache_step(moe_cache * mc);

// tear down the cache and stop the upload worker thread; safe to call more than once
void llama_moe_cache_destroy(moe_cache * mc);
