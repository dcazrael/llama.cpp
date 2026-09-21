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
idx_scratch_target = 512 MiB
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
