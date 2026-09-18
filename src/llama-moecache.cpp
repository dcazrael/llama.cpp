#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct layer_state {
    llama_moe_cache_layer pub;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;    // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;    // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use;  // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;        // uncached ids observed since last step (dedup)

    std::vector<bool>     slot_in_flight; // slot has an upload pending

    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
};

struct moe_cache_impl {
    int32_t n_slots     = 0;
    int32_t max_inserts = 2;

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;
};

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, void * ud) {
    moe_cache_impl * mc = reinterpret_cast<moe_cache_impl *>(ud);

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens > 4) {
        return; // batch/prefill: the cache graph is not built there, don't pollute the LRU
    }

    // "blk.<il>.ffn_gate_exps.weight"
    if (strncmp(name, "blk.", 4) != 0) {
        return;
    }
    const int il = atoi(name + 4);

    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (!ls) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit++;
                ls->slot_last_use[slot] = ++mc->clock;
            } else {
                ls->n_miss++;
                bool dup = false;
                for (int32_t p : ls->pending) {
                    if (p == id) { dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                }
            }
        }
    }
}

void upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) ||
        (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu "
                        "dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz,
                dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    ggml_backend_tensor_set(dst_c,
            (const char *) src->data + (size_t) expert*sz,
            (size_t) slot*dst_c->nb[2], sz);
}

void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    ggml_backend_tensor_set(pub.dev_table,  &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
    ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
}

// cast opaque moe_cache* to the internal implementation type
static inline moe_cache_impl * impl(moe_cache * mc) {
    return reinterpret_cast<moe_cache_impl *>(mc);
}
static inline const moe_cache_impl * impl(const moe_cache * mc) {
    return reinterpret_cast<const moe_cache_impl *>(mc);
}

} // namespace

int llama_moe_cache_create(moe_cache ** out, const llama_model & model, int32_t n_slots, int32_t max_inserts) {
    if (*out || n_slots <= 0) {
        return 0;
    }

    auto * mc = new moe_cache_impl();
    mc->n_slots = n_slots;
    if (max_inserts > 0) {
        mc->max_inserts = max_inserts;
    }

    // collect the host-resident expert layers, grouped by the device buffer
    // type of that layer's router (the cache lives next to the router)
    struct cand { int il; const llama_layer * l; };
    std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
            continue;
        }
        if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
            continue; // dry-run / memory-estimation model: weights not loaded
        }
        if (!l.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
            continue; // experts already on a device: nothing to cache
        }
        if (!l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
            continue; // no device home for the cache
        }
        groups[ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)].push_back({(int) il, &l});
    }

    if (groups.empty()) {
        LLAMA_LOG_INFO("%s: n_moe_cache_slots=%d but no host-resident expert layers found - disabled\n",
                __func__, n_slots);
        delete mc;
        return 0;
    }

    // host buffer for the CPU-side tables
    std::vector<cand> all;
    for (auto & g : groups) {
        all.insert(all.end(), g.second.begin(), g.second.end());
    }

    auto alloc_group = [&](ggml_backend_buffer_type_t buft,
                            const std::vector<cand> & cands, bool tables_only) -> bool {
        ggml_init_params ip = {
            /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*4 + 8),
            /*.mem_buffer=*/ nullptr,
            /*.no_alloc  =*/ true,
        };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            return false;
        }
        mc->ctxs.push_back(ctx);

        for (const auto & c : cands) {
            layer_state * ls = nullptr;
            for (auto & l : mc->layers) {
                if (l.pub.il == c.il) { ls = &l; break; }
            }
            if (!ls) {
                mc->layers.push_back({});
                ls = &mc->layers.back();
                ls->pub.il       = c.il;
                ls->pub.n_slots  = n_slots;
                ls->pub.up_src   = c.l->ffn_up_exps;
                ls->pub.gate_src = c.l->ffn_gate_exps;
                ls->pub.down_src = c.l->ffn_down_exps;
            }

            if (tables_only) {
                ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
            } else {
                const ggml_tensor * u  = c.l->ffn_up_exps;
                const ggml_tensor * g  = c.l->ffn_gate_exps;
                const ggml_tensor * d  = c.l->ffn_down_exps;
                ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], n_slots + 1);
                ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], n_slots + 1);
                ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], n_slots + 1);
                ls->pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
                ggml_format_name(ls->pub.up_c,      "moe_cache_up.%d",   c.il);
                ggml_format_name(ls->pub.gate_c,    "moe_cache_gate.%d", c.il);
                ggml_format_name(ls->pub.down_c,    "moe_cache_down.%d", c.il);
                ggml_format_name(ls->pub.dev_table, "moe_cache_tbl.%d",  c.il);
            }
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                    __func__, ggml_backend_buft_name(buft));
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        mc->bufs.push_back(buf);
        return true;
    };

    bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
    for (auto & g : groups) {
        if (!ok) {
            break;
        }
        ok = alloc_group(g.first, g.second, /*tables_only=*/false);
    }

    if (!ok) {
        for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
        for (auto * c : mc->ctxs) { ggml_free(c); }
        delete mc;
        return 0; // allocation failed: stay disabled
    }

    // init LRU state + tables (everything uncached -> dummy slot n_slots)
    size_t vram = 0;
    for (auto & ls : mc->layers) {
        const int64_t n_expert = ls.pub.up_src->ne[2];
        ls.slot_expert.assign(n_slots, -1);
        ls.expert_slot.assign(n_expert, -1);
        ls.slot_last_use.assign(n_slots, 0);
        ls.slot_in_flight.assign(n_slots, false);

        std::vector<int32_t> dummy(n_expert, n_slots);
        ggml_backend_tensor_set(ls.pub.dev_table,  dummy.data(), 0, n_expert*sizeof(int32_t));
        ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));

        mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
        vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
        LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert\n",
                ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2]);
    }

    mc->worker = std::thread([mc]() {
        for (;;) {
            upload_job j;
            {
                std::unique_lock<std::mutex> lk(mc->wmtx);
                mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                if (mc->stop) {
                    return;
                }
                j = mc->todo.front();
                mc->todo.pop_front();
            }
            auto & ls = mc->layers[j.layer_idx];
            upload_slice(ls.pub.up_c,   ls.pub.up_src,   j.expert, j.slot);
            upload_slice(ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot);
            upload_slice(ls.pub.down_c, ls.pub.down_src, j.expert, j.slot);
            {
                std::lock_guard<std::mutex> lk(mc->wmtx);
                j.done = true;
                mc->done.push_back(j);
            }
        }
    });

    ggml_set_moe_obs_callback(moe_obs_cb, mc);
    *out = static_cast<moe_cache *>(static_cast<void *>(mc));

    LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu layers x %d slots, %d inserts/step, %.1f MiB device memory\n",
            __func__, mc->layers.size(), n_slots, mc->max_inserts, vram/1024.0/1024.0);
    return 0;
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const moe_cache * mc, const ggml_tensor * up_exps) {
    if (!mc) {
        return nullptr;
    }
    const moe_cache_impl * imp = impl(mc);
    auto it = imp->by_up_src.find(up_exps);
    if (it == imp->by_up_src.end()) {
        return nullptr;
    }
    return &imp->layers[it->second].pub;
}

void llama_moe_cache_step(moe_cache * mc) {
    moe_cache_impl * imp = impl(mc);
    // 1) publish completed uploads (sync point: synchronize was called before step)
    {
        std::lock_guard<std::mutex> wlk(imp->wmtx);
        std::lock_guard<std::mutex> lk(imp->mtx);
        for (const auto & j : imp->done) {
            auto & ls = imp->layers[j.layer_idx];
            ls.slot_expert[j.slot]     = j.expert;
            ls.expert_slot[j.expert]   = j.slot;
            ls.slot_last_use[j.slot]   = ++imp->clock;
            ls.slot_in_flight[j.slot]  = false;
            set_table_entry(ls.pub, j.expert, j.slot);
        }
        imp->done.clear();
    }

    std::lock_guard<std::mutex> lock(imp->mtx);
    imp->n_steps++;

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    for (size_t li = 0; li < imp->layers.size(); ++li) {
        auto & ls = imp->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        int budget = imp->max_inserts;
        for (auto it = ls.pending.rbegin(); it != ls.pending.rend() && budget > 0; ++it, --budget) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0) {
                continue;
            }

            // victim: an empty non-in-flight slot if any, else the LRU non-in-flight slot
            int32_t slot = -1;
            uint64_t best = UINT64_MAX;
            for (int32_t s = 0; s < imp->n_slots; ++s) {
                if (ls.slot_in_flight[s]) {
                    continue;
                }
                if (ls.slot_expert[s] < 0) { slot = s; break; }
                if (ls.slot_last_use[s] < best) { best = ls.slot_last_use[s]; slot = s; }
            }
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(ls.pub, victim, imp->n_slots);
            }
            ls.slot_in_flight[slot] = true;

            std::lock_guard<std::mutex> wlk(imp->wmtx);
            imp->todo.push_back({li, id, slot});
        }
        ls.pending.clear();
    }
    imp->wcv.notify_one();

    if (imp->n_steps % 512 == 0) {
        uint64_t h = 0, m = 0;
        for (auto & ls : imp->layers) { h += ls.n_hit; m += ls.n_miss; }
        LLAMA_LOG_DEBUG("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%%\n",
                imp->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0);
    }
}

// tear down the cache and stop the upload worker thread; safe to call more than once
void llama_moe_cache_destroy(moe_cache * mc) {
    if (!mc) {
        return;
    }
    moe_cache_impl * imp = impl(mc);

    // signal the worker to stop
    {
        std::lock_guard<std::mutex> lk(imp->wmtx);
        imp->stop = true;
    }
    imp->wcv.notify_one();
    if (imp->worker.joinable()) {
        imp->worker.join();
    }

    // free device buffers first (so tensors are no longer allocated while contexts are freed)
    for (auto * b : imp->bufs) { ggml_backend_buffer_free(b); }
    for (auto * c : imp->ctxs) { ggml_free(c); }

    ggml_set_moe_obs_callback(nullptr, nullptr);
    delete imp;
}
