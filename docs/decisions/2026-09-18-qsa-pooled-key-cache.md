# 2026-09-18 Qwen4Exp Incremental Pooled-Key Cache

## Goal

Avoid recomputing all historical pooled-key summaries every decode token
in Qwen3.8-Flash-Next (`qwen4exp`). The pooled cache keeps one f32 row per
position block, one row per QSA layer, so only newly completed blocks need
mean-pool, norm and rope per step.

This is "Phase 10 - Incremental pooled-key cache" in
`docs/dual-gpu-optimization-plan.md` and the third Flash-Next-specific
optimization added to the dual-GPU fork.

## Decision

Incremental pooled-key cache, per-device buffer allocation, dirty-table
tracking via `set_input_qsa`, watermark-based validity tied to sequence
mutations (seq_rm / seq_cp / seq_add / seq_div / clear / state load / state
drop).

### Kill switch

`LLAMA_QSA_NO_POOLED_CACHE=1` disables the pooled path entirely and falls
back to full recompute of blk_cells / blk_pos.

### Per-device allocation

One buffer per `ggml_backend_buffer_type_t`, not a single global buffer.
On layer-split setups each GPU owns specific layers and a pooled buffer
that lives on the wrong GPU would cross the inter-GPU link every decode
step.

### Multi-stream

Single-stream memories only. A unified cache shares one stream but block
rows are position-indexed; multi-stream memories fall back to full
recompute without warning.

### Dirty table approach

`set_input_qsa` resolves which blocks the graph must (re)pool this ubatch
and advances the watermark. Complete blocks are immutable so rows below
the watermark stay valid. Incomplete blocks are masked by a stale row
bias (a full row of -inf never produces a valid attention score).

### State restore semantics

A `state_read` (full state load) invalidates the pooled cache so the
graph recomputes everything. A `state_drop` (restore failure or full
context wipe) drops pooled rows just like any other cache state.

Partial state restore (PARTIAL_ONLY, cells unchanged) does not invalidate
pooled cache. Rollbacks arrive as seq_rm, which clamps the watermark
before the next set_input_qsa call.

## What changed

### Header

`src/llama-memory-hybrid-idx.h`: pooled cache fields on the class.

- `pooled_ctxs` / `pooled_bufs`: one ggml_context / buffer per device
- `pooled_k`: map from layer index to the pooled tensor
- `pooled_rows` / `pooled_ratio`: row count and ratio for the layer
- `pooled_w`: per-sequence watermark (block index)
- `get_pooled_k()` / `get_pooled_rows()` / `pooled_valid()` accessors

### Implementation

`src/llama-memory-hybrid-idx.cpp`:

- Constructor allocates pooled buffers (one per device, skip recurrent
  layers, only allocate for dense-attention layers with ratio > 0).
- `set_input_qsa` fills dirty tables (`dirty_cells` / `dirty_pos` /
  `dirty_rows`) when the pooled cache is active. Each dirty row points
  at a complete position block; the graph gathers, normalises, applies
  rope and writes the row via set_rows.
- Sequence mutation hooks update the watermark:
  - `pooled_rm`: clamp to the earliest surviving position block
  - `pooled_reset`: reset every sequence's watermark
  - `seq_cp`: reset the destination sequence's watermark (new context)
  - `seq_div`: clamp proportional to the division factor
  - `clear`: reset all watermarks
  - `state_read`: reset all watermarks
  - `state_drop`: reset all watermarks

### Graph building

`src/models/qwen4exp.cpp`: when `dirty_cells` / `dirty_pos` /
`dirty_rows` are provided the pooled path builds a separate gather ->
layer_norm -> rope -> set_rows subgraph. When they are null the graph
still builds the full blk_cells / blk_pos path (full recompute).

### Fallback regression fix (commit `fe41525eb`)

The pooled cache refactoring replaced direct writes to `dst_blk_cells` /
`dst_blk_pos` with per-call `loc_blk_cells` / `loc_blk_pos` temporary
vectors and copies. This added O(r * n_blocks) allocation and copy every
decode step even on the fallback path, degrading throughput from ~7.16
t/s to ~5.55 t/s.

The fix adds a fast-fallback branch in `set_input_qsa` that writes
directly to destination buffers when `dirty_cells == nullptr`, matching
the pre-pooled-cached code's behaviour and restoring the ~7-8 t/s range.

## Benchmark results (Maya: dual RTX 3060)

Qwen3.8-Flash-Next, long-context TG benchmark.

### 34k context

```
Pooled cache ON:  9.06 t/s
Pooled cache OFF: 8.80 t/s
Improvement:      ~+3.0%
```

### 49k context

```
Pooled cache ON:  8.40 t/s
Pooled cache OFF: 8.08 t/s
Improvement:      ~+4.0%
```

The pooled cache provides a consistent ~3-4% decode throughput gain at
deep context. The gain scales with context length because more historical
blocks can be served from the cache.

### Regression history

Before the fallback fix (`fe41525eb`), Task 3 base (pooled OFF) measured
~5.55 t/s at the same context depth. The pooled ON path also measured
~5.57 t/s. Both paths were equally degraded because the regression was in
the shared set_input_qsa path, not the pooled cache itself.

The fix restores pooled OFF to the expected ~7-9 t/s range while keeping
the pooled ON gain intact.

## Tests

`test-llama-archs --arch qwen4exp` passes on both RTX 5070 and CPU with
pooled ON (NMSE ~8.6e-08) and OFF (NMSE ~8.7e-08).

## Commits on dual-gpu-opt

- `b3e46da99` qwen4exp: cache QSA pooled keys incrementally
- `c68829505` llama: fix QSA set_input_qsa fallback overhead
