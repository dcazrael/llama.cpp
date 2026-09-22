# 2026-09-17 Qwen4Exp QSA Indexer Token Chunking

## Goal

Reduce peak graph scratch / workspace memory for the Qwen3.8-Flash-Next
(`qwen4exp`) QSA indexer during prompt processing at large contexts, on the
dual RTX 3060 daily driver.

This is the first Flash-Next-specific optimization in the dual-GPU fork and is
the implementation of "Phase 8 - QSA indexer scratch chunking" in
`docs/dual-gpu-optimization-plan.md`.

## Decision

Process the QSA indexer score computation in bounded chunks along the token
axis. Keep all other QSA work unchanged. The implementation is narrow: it lives
entirely inside `llama_model_qwen4exp::graph::build_qsa_top_k` in
`src/models/qwen4exp.cpp`.

## What changed

`src/models/qwen4exp.cpp`, in `build_qsa_top_k`. The per-token computation
between `cb(q, "indexer_q", il)` and `return top_k;` was wrapped in a chunk
loop. The shared work (key projection, pooled keys, RoPE, q projection + RoPE)
stays outside the loop. Each chunk computes:

1. a `q_c` view into the per-token query tensor,
2. a `bias_c` view into the per-token bias (block-level or cell-level depending
   on the `blk_bias` branch),
3. optionally a `mask_c` view into the kq mask (only when `blk_bias` is true).
   When the source KQ mask is F16 (flash attention), the cast to F32 is taken
   on the chunk slice, not on the full mask, so a chunked execution does not
   materialise a full-token F32 mask cast before taking the slice.
4. the score GEMM, the head reduction, the block-bias add, the per-cell
   expansion, the cell-bias / mask add, and the top-k,
5. concat the chunk's top-k onto the running accumulator along the token axis.

The accumulator starts as the first chunk's top-k and grows by concatenation.

The original (unchunked) graph shape, ordering, types and cb names are preserved
when `nc == n_tps` (the loop runs exactly once). For larger ubatches the loop
runs multiple times, and `cb` calls are skipped because an eval-callback dump
would otherwise produce one identically-named tensor per chunk per layer.

## Why chunking reduces peak scratch

In the unchunked path the dominant scratch tensors are:

- the per-token query `q` (`[idx_dim, n_idx_h, n_tokens]` F32),
- the mul_mat output `score` (`[n_blocks, n_idx_h, n_tps, n_stream]` F32),
- the head-reduction accumulator `summed` (`[n_blocks, n_tps, n_stream]` F32),
- the per-cell `expanded` (`[n_kv, n_tps, n_stream]` F32),
- the second `cont(permute(expanded))` (`[n_kv, n_tps, n_stream, 1]` F32),
- the cell-bias / mask add output (`[n_kv, n_tps, n_stream, 1]` F32),
- the optional F32 cast of the KQ mask (`[n_kv, n_tps, n_stream]` F32, only when
  flash attention keeps the KQ mask in F16).

The three `[n_kv, _, n_stream]` F32 tensors in the expanded family all scale
linearly with `n_tokens` and dominate the per-device scratch at large prefill
depths. They scale together (each at `n_kv * n_tokens` F32) and together cap
the usable ubatch before either the allocator or the device's VRAM budget
gives up.

No reduction in this path crosses tokens: each token's top-k depends only on
its own query row, the shared `pooled` keys, and the shared `cell_blk` /
`bias` / kq_mask (sliced per chunk). The token loop can therefore be split into
chunks with identical results, and `ggml-alloc` reuses one buffer across the
chunks. The peak scratch is bounded by the chunk size rather than by the ubatch.

## Reference implementation

Commit `dba3221199c516b588c913cebaad4e3fec364bdc` in `ggml-org/llama.cpp`
("`llama : chunk the glm5next DSA indexer over tokens`"). That commit studies
the same problem on GLM-5.3-Flash's DSA indexer. The mechanics it uses (chunk
along the token axis, slice `q` / `bias` with views, append the per-chunk
top-k via `ggml_concat` on the token axis, keep `cb` only for the unchunked
case, derive the chunk size from a fixed scratch target) are reused here.

The Qwen4Exp indexer differs from GLM's DSA in two ways that this code has to
adapt for:

- The head reduction is a Python-style `ggml_add` accumulation (n_idx_h adds)
  rather than `ggml_sum_rows`. This makes the per-chunk scratch scale with
  `n_idx_h` and not with a separate `materialised twice` factor.
- The top-k is over per-cell scores (`expanded = get_rows(score, cell_blk)`),
  not over per-pool scores followed by an explicit pool-to-cell expansion.

Both differences are accommodated by replacing `n_tps` with `nc` inside the
loop and slicing `bias`, `kq_mask` and `q` accordingly. The chunked algorithm
is the unchunked algorithm applied `n_chunks = ceil(n_tps / idx_chunk)` times
in order.

## Chunk-size behaviour

```text
idx_overhead       = 2 + (blk_bias && kq_mask is not F32 ? 1 : 0)
idx_bytes_per_step = idx_overhead * n_kv * n_stream * sizeof(float)
idx_scratch_target = LLAMA_QSA_SCRATCH_MIB (default 512 MiB)
idx_chunk          = clamp(idx_scratch_target / idx_bytes_per_step, 1, n_tps)
```

The overhead counts the largest simultaneously alive F32 tensors in this
path. Three copies of `[n_kv, nc, n_stream]` F32 coexist at the final
bias/mask add: the permute+cont output, the bias/mask input, and the add
output. The mask input is an F32 cast only when `blk_bias` is set and the
source KQ mask is not already F32 (flash attention keeps the KQ mask in
F16). The head-reduction and `score` temporaries are smaller (`[n_blocks,
nc, n_stream]` and `[n_blocks, n_idx_h, nc, n_stream]` F32 respectively)
and do not overlap the expanded tensors in lifetime, so they don't add to
the per-chunk peak. `n_stream` is in the divisor because one chunk step
covers `nc * n_stream` tokens.

Short contexts where the per-token scratch already fits the target come out
unchunked (`idx_chunk == n_tps`, single iteration) and take the previous code
path unchanged. For the `n_kv = 131072, n_stream = 2, blk_bias F16`
example (the case the prior version of this doc stated gave ~1024):

- `idx_overhead = 3`
- `idx_bytes_per_step = 3 * 131072 * 2 * 4 B = 3 MiB`
- `idx_chunk = clamp(512 MiB / 3 MiB, 1, n_tps) = clamp(~170, 1, n_tps) = 170`

For non-`blk_bias` (or F32 mask) at the same `n_kv` and `n_stream`:

- `idx_overhead = 2`
- `idx_chunk = clamp(512 MiB / 2 MiB, 1, n_tps) = 256`

The final chunk can be partial: `nc = min(idx_chunk, n_tps - t0)`. The same
ops run with the smaller `nc`, the top-k concatenates onto the accumulator,
and the output shape is identical to the unchunked case
(`[width, n_tps, 1, n_stream]` I32).

## Important assumptions and limitations

1. The implementation does not depend on exactly two GPUs, RTX 3060, CUDA
   device numbering, or the current tensor-split ratio. It is a generic
   Qwen4Exp graph optimisation that any backend running the same model
   benefits from.

2. The chunk size is derived from a fixed scratch target, not from the device
   VRAM. Choosing a target small enough to leave room on ~8 GB devices keeps
   the per-chunk scratch bounded even on memory-tight hardware.

3. The per-chunk tensor count is proportional to `n_idx_h` because the head
   reduction accumulates `n_idx_h` `ggml_add` operations. On models with a
   very large `indexer_n_head` combined with a very small scratch target, the
   total graph node count grows linearly with the number of chunks. The
   production target (typical `indexer_n_head` is small) does not hit this,
   but a stress-test scenario with extreme values can. This is a degenerate
   case for which chunking is the wrong tool; the correct fix is to make the
   per-iteration tensor count independent of `n_idx_h`, which is out of scope
   for this task.

4. The accumulator `top_k` grows by `ggml_concat` along the token axis. Each
   concat re-copies earlier chunks' indices, which is negligible I32 traffic
   next to the scoring GEMMs but does grow the accumulator linearly with
   `n_tps`. The final `top_k` is small (`[width, n_tps, 1, n_stream]` I32,
   width ~ `indexer_top_k + r - 1`).

5. The change is confined to `src/models/qwen4exp.cpp`. No CLI option, no
   graph-reserve change, no other source file is modified. Validation is via
   `test-recurrent-state-rollback` on the qwen4exp test model at multiple
   context sizes; the unchunked path is exercised at the production target
   and a moderately chunked path is exercised at a smaller scratch target.
   Both produce bit-identical logits to the unchunked graph at matched
   ubatch.

## Maya deep-context tuning update (2026-09-22)

The initial 2 GiB target was too permissive for the dual RTX 3060 daily-driver
configuration. At 128K context / ubatch 8192 the full prompt-processing graph
still requested multi-GiB device-local workspaces and OOMed on CUDA0. A
50/50 -> 45/55 CUDA0/CUDA1 split reduced the failed CUDA0 workspace request
from about 8.54 GiB to 7.71 GiB, proving placement helps, but not enough by
itself.

The target is therefore reduced to 512 MiB for the next validation pass. This
is an intentionally conservative capacity test: if 128K / ubatch 8192 fits,
the target can later be tuned upward for prompt-processing throughput. If it
still OOMs, the next memory wall to address is the dense QSA attention-mask
materialization in build_attn_qsa(), not a further reduction in ubatch.


## Attention-side QSA chunking (2026-09-22)

The indexer-score chunking experiment proved that bounding only the indexer
scratch does not control the deep-context peak. On Maya, lowering the score
scratch target from 512 MiB to 64 MiB left the fatal ~6 GiB-class compute
workspace essentially unchanged.

The branch now ports the bounded QSA prefill design from upstream experimental
commit 8a0ad638, with the later causal-selection fix from 4c552387 and the
Generel sampled-decode behavior preserved.

For single-stream flash-attention QSA, the graph now:
- builds pooled indexer state once,
- processes query tokens in adaptive 64..256-token QSA chunks,
- scores/top-k selects each chunk independently,
- rebuilds the causal mask from cache cell positions on device,
- materializes only an [n_kv, n_chunk] mask,
- runs sparse flash attention for that query chunk,
- scatters each chunk output back into the outer ubatch output.

The outer llama ubatch remains unchanged. The goal is therefore still
~128K active context with ubatch 8192; only the QSA attention subgraph is
microbatched internally.

The old LLAMA_QSA_SCRATCH_MIB score-target sweep is retired from the active
path. If the 128K/ub8192 target still OOMs after this change, investigate the
next device-local peak (notably quantized-K/V F16 staging in flash attention)
rather than returning to score-scratch tuning or low-ubatch boundary mapping.


## Block-granular top-k follow-up (2026-09-22)

The chunked attention path originally still expanded every block score back to
all `n_kv` cells and ran top-k over that `[n_kv, n_chunk]` tensor. That work
is redundant for Qwen4Exp: a compressed block has one indexer score and the
attention budget is defined in whole blocks, with only the incomplete causal
tail handled separately.

The branch now ports the block-selection idea from the Qwen4Exp work discussed
in ggml-org/llama.cpp#28734:

- when block bias is valid, rank `n_blocks ~= n_kv/compress_ratio` directly,
- expand only the selected blocks through `blk_cells`,
- carry the incomplete tail in a small `extra_cells[ratio, n_tokens]` input,
- mark fully-future blocks as `-inf` before block selection,
- keep the existing per-cell causal mask authoritative after selection,
- retain the old token-level selection path for layouts where block bias is not
  valid.

This applies to both the normal QSA path and the active 64..256-token chunked
prefill path. It removes the n_kv-wide F32 score expansion and n_kv-wide top-k
sort from the normal causal path without changing the outer llama ubatch.

This does **not** yet remove the chunk-sized `[n_kv, n_chunk]` F16 attention
mask. Sparse Flash Attention still receives a mask and compacts it back into
indices internally. A direct selected-index FA interface, or an equivalent
compact gather, remains a separate optimization angle if the 128K/ub8192 target
still lacks workspace headroom.


## Block-first graph-input allocation fix (2026-09-22)

The first block-granular top-k integration still created `cell_blk` for the
block-bias path even though no graph node consumed it anymore. ggml therefore
did not allocate a backend buffer for that input. During runtime,
`set_input_qsa()` called `ggml_backend_buffer_is_host(cell_blk->buffer)`, which
reached `ggml_backend_buffer_get_usage()` with a null buffer and aborted at
`ggml-backend.cpp:205: GGML_ASSERT(buffer)`.

The graph-input contract is now explicit:

- `cell_blk` is created only for the fallback token-level selection path,
- `cell_pos` is created only when device-side causal reconstruction consumes it,
- `set_input_qsa()` accepts either pointer as null,
- `n_kv` is derived from the indexer cache instead of `cell_blk`,
- writes into the optional cell map are guarded.

This keeps the block-first path free of unused n_kv-wide inputs and avoids
backend-buffer assertions during graph execution.


## Cache-width API correction (2026-09-22)

The first null-buffer fix accidentally called `llama_kv_cache::get_n_kv()`
without the required slot-info argument and therefore did not compile. The
correct layering matches the upstream QSA branch: the hybrid indexer context
owns a `llama_kv_cache_context`, obtains the active width with
`get_idx()->get_n_kv()`, and forwards that width explicitly to the lower-level
`llama_memory_hybrid_idx::set_input_qsa()` helper.

This is a plumbing fix only. It does not revert or weaken native q4 Flash
Attention, block-first QSA selection, chunked QSA prefill, or sparse Flash
Attention.


## Phase-aware prefill residency follow-up (2026-09-22)

The first successful ISTA GSQ-RCO IQ3_XXS 128K/ub8192 run processed 16384
prompt tokens at about 484 -> 462 tok/s, then failed before the third chunk
while trying to reserve a 5907.82 MiB shared CUDA1 compute workspace.

Two independent sources of avoidable pressure were present:

1. live-context workspace growth rounded the required ~24576 KV cells to a
   32768-cell reserve graph, even though reservation is revisited each prompt
   chunk;
2. the configured 48-slot MoE cache was aggressively filling during prefill
   (36.63% L1 hit rate, 13277 misses, ~7.3 GiB demand H2D traffic) while the
   prefill workspace needed its largest arena.

The branch now uses 1024-cell live-KV reserve alignment instead of power-of-two
growth. With phase-aware workspace enabled, the backend candidate snapshot is
also capped to 16 slots during prefill. Candidate replacement already retires
and frees grouped/legacy CUDA cache resources, so phase changes provide real
VRAM headroom. The model configuration remains cache48; decode restores the
configured slot count after the prompt workspace contracts.

This does not change the outer ubatch, native-q4 FA, block-first QSA selection,
or sparse FA. The residual [n_kv, n_chunk] F16 sparse-attention mask remains a
separate deeper optimization if this headroom is still insufficient.
