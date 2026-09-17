# 2026-09-17 Qwen4Exp QSA Sparse Flash Attention

## Goal

Bound attention work on Qwen3.8-Flash-Next (`qwen4exp`) by the QSA selection
width instead of rescoring/masking the full KV cache at deep context.

This is "Phase 9 - QSA sparse Flash Attention" in
`docs/dual-gpu-optimization-plan.md` and the second Flash-Next-specific
optimization added to the dual-GPU fork.

## Decision

Port upstream PR #28770 (`CUDA: enable sparse fa for qwen4`, head
`41a4ad00dd35fc096f2de4e41549baee2fef6f23`) onto the current codebase, and
uncomment the qwen4exp `build_attn_mha` call so it actually carries the QSA
top-k width as `n_kv_max`. The CUDA-side changes extend the existing sparse
Flash Attention path to cover qwen4's `head_dim = 256`, both `ncols1 = 1`
(decode) and `ncols1 = 8` (prefill).

The dense/masked fallback is preserved. The model-side change is a single line;
the CUDA-side change widens the per-tile index gather from a single query to a
group of `ncols1` queries sharing one index list.

## What changed

### Model side

`src/models/qwen4exp.cpp`, in `build_attn_qsa`. The TODO that was holding the
sparse call out is replaced by the actual call, with `top_k->ne[0]` (the QSA
top-k width for this layer) passed as the `n_kv_max` parameter of
`llm_graph_context::build_attn_mha`:

```cpp
ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
```

The width is `min(n_kv, indexer_top_k + r - 1)` where `r` is the layer's
`dsv4_compress_ratios[il]`. The actual runtime value of `n_kv_max` is whatever
the model's GGUF metadata says, and is logged at the kernel-selection point
(see Observability).

### CUDA side

`ggml/src/ggml-cuda/fattn-common.cuh`, `ggml/src/ggml-cuda/fattn-mma-f16.cuh`,
`ggml/src/ggml-cuda/fattn.cu`:

- `ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse` now also accepts
  `DKQ == DV == 256` with `(ncols1, ncols2)` in `{(1, 8), (8, 8)}`. Without
  this the sparse FA kernel is never selected for qwen4's head shape.
- `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` takes `cc` and `ncols1`
  as explicit parameters. The `K->ne[1] >= ...` guard scales with the gather
  work for the ncols1 value the selector passes in: `2 * n_gather` where
  `n_gather = (ncols1 == 1 ? Q->ne[1] : ncols1) * n_kv_max`. This replaces the
  per-query `2 * n_kv_max` threshold so the dense fallback remains the right
  call for short prefill where the sparse gather would not actually save work.
- `flash_attn_mask_to_sparse_indices` (in `fattn.cu`) now extracts one index
  list per group of `ncols1` queries, taking the OR of the queries' visibility
  masks. The live count of each list is written to a parallel `counts` array
  allocated right after the indices.
- `flash_attn_ext_f16` (in `fattn-mma-f16.cuh`) reads `iter_j` rather than
  `ne31` for the per-query offset so the OR-reduced lists stay aligned with
  `ncols1 > 1` tiles. The kernel uses the `counts` array to bound the inner
  iteration: `kb0_stop = ceil(live_count / nbatch_fa)`. Without the bound,
  tiles would iterate past the live indices into uninitialised storage.
- The kernel selector excludes the vec path when a GQA-sparse decode applies,
  so the sparse gather cannot be silently substituted by the vec kernel.

### Tests

`tests/test-backend-ops.cpp` gains three `test_flash_attn_ext` cases for
qwen4's shape (`DKQ = DV = 256`, GQA = 12): single-token decode, batched
prefill on a single stream, and batched prefill split across two streams.

### Observability

Two verbose-only (`GGML_LOG_DEBUG` / `LLAMA_LOG_DEBUG`) one-shot log lines
let a benchmark confirm the wiring without changing default output:

1. `src/llama-graph.cpp` in `llm_graph_context::build_attn_mha`. Fires the
   first time the model passes a non-zero `n_kv_max` into the `flash_attn_ext`
   op. The message makes clear that it only confirms model-side wiring, not
   that the backend actually selected the sparse kernel:

   ```
   flash_attn_ext op received n_kv_max=<N> from the model
   ```

2. `ggml/src/ggml-cuda/fattn-mma-f16.cuh` in
   `ggml_cuda_flash_attn_ext_mma_f16_case`, right after `use_sparse = true` is
   set. Fires once per `(DKQ, DV, ncols1, ncols2)` specialisation, and
   reports everything needed to verify the selected path:

   ```
   sparse flash attention: device=<id> cc=<major>.<minor> DKQ=<DKQ> DV=<DV> ncols1=<n> ncols2=<n> Q_tokens=<n> K_len=<n> n_kv_max=<n> sparse=true
   ```

   This message is the proof that the CUDA backend actually picked the sparse
   kernel; the graph-side message above is not.

Both messages are gated on `LOG_LEVEL_DEBUG` (verbosity 5+) so they do not
appear in default log output. To inspect them during Maya benchmarking, run
with `-v 5` (or set `--log-level debug`).

## How the sparse bound is calculated

For each attention layer with `r = dsv4_compress_ratios[il] > 0`, the indexer
picks the top `width = min(n_kv, indexer_top_k + r - 1)` cells per query
(whole `r`-sized blocks plus a possible tail). The mask build in
`build_attn_qsa` zeroes every other K position. The width is then passed as
`n_kv_max` to the flash attention op. The CUDA kernel stops its K iteration at
the live count of the sparse index list, so the per-tile work is bounded by
`width` instead of `n_kv`.

The exact `n_kv_max` value the production model uses comes from its GGUF
metadata and is not hardcoded. The runtime diagnostic line exposes it for
each specialisation; rely on that rather than guessing.

## Threshold and selector flow

The CUDA sparse path goes through two `shall_use_sparse` checks. Tracing the
actual code in `ggml/src/ggml-cuda/fattn.cu` and `fattn-mma-f16.cuh`:

1. **Early check in `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV,
   ncols2>`.** When `may_use_sparse(DKQ, DV, 1, ncols2)` is true (qwen4's
   `(256, 256, 1, 8)` case), the dispatcher tries the sparse path first with
   `ncols1 = 1`:

   ```
   n_gather_early = Q->ne[1] * n_kv_max
   threshold_early: K->ne[1] >= max(4096, 2 * n_gather_early)
   ```

   If this passes, the dispatcher calls `case<DKQ, DV, 1, ncols2>` and
   returns. For qwen4 with `ncols2 = 8` and `n_kv_max` ~5xx, decode
   (`Q->ne[1] = 1`) clears the threshold trivially and the path is taken.
   For deep prefill (`Q->ne[1] = 64`) the threshold becomes roughly
   `K->ne[1] >= 64 * n_kv_max` and only fires at very deep prefill.

2. **Normal ncols1 selection** in the same function. When the early sparse
   check fails (or does not apply), the dispatcher falls through to the
   standard ncols1 ladder (`8/ncols2`, `16/ncols2`, `32/ncols2`, `64/ncols2`)
   until one fits `Q->ne[1]`. For qwen4 with `ncols2 = 8` this ladder always
   picks `ncols1 = 1` (because `8/ncols2 = 1`) when `Q->ne[1] <= 1`, and
   `ncols1 = 8` (from the `64/ncols2` terminal case) otherwise.

3. **Inner check in `ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, ncols1,
   ncols2>`.** Whatever `ncols1` was chosen above, the case function calls
   `shall_use_sparse(cc, dst, ncols1)` again. With `ncols1 > 1`:

   ```
   n_gather_inner = ncols1 * n_kv_max
   threshold_inner: K->ne[1] >= max(4096, 2 * n_gather_inner)
   ```

   For qwen4 prefill with `ncols1 = 8`, `n_gather = 8 * n_kv_max` and the
   threshold is `K->ne[1] >= 16 * n_kv_max`. This is much smaller than the
   per-sequence `2 * Q->ne[1] * n_kv_max` that the early check uses, which is
   why prefill typically takes the dense fallback for short contexts and the
   sparse path once the cache is deep enough to make the gather worthwhile.

When any condition fails, the dense/masked FA path runs.

The runtime log line reports the exact `(DKQ, DV, ncols1, ncols2, Q_tokens,
K_len, n_kv_max)` tuple that won. Read those numbers, not a hard-coded
assumption, to determine which threshold applied.

## Correctness invariants

- The sparse bound never excludes a token that the indexer selected. The mask
  build in `build_attn_qsa` unmasks exactly the cells named by `top_k` and the
  kernel reads only those.
- The causal/visibility semantics of the KQ mask are unchanged: the sparse
  gather applies on top of the per-position mask that the regular attention
  already uses.
- The dense fallback produces equivalent logical results because the mask
  build already implements the "selected cells" semantics in dense form.
- The per-call scratch buffer is bounded by
  `n_kv_max * n_lists + n_lists` where `n_lists = ntiles_x * mask->ne[3]`.
  There is no dynamic allocation inside the kernel.
- No new cross-device transfers: the indices and counts are scratch buffers
  that live on the same device as the FA op.

## Validation performed

Local validation was done on the RTX 5070 development machine used for
incremental work in this fork (`compute capability 12.0`, SM120, 11 810 MiB
VRAM). Target SM86 validation on the dual RTX 3060 daily driver (Maya) is
out of scope for this task and will be done separately as a benchmark pass.

- `cmake --build . --target ggml-cuda llama test-backend-ops test-llama-archs`
  builds clean with CUDA 13.4 / driver 615.71.
- `./bin/test-backend-ops -o FLASH_ATTN_EXT`: 3982/3982 tests pass on the
  RTX 5070, including the three new qwen4-shape sparse FA cases and the
  pre-existing sparse FA tests for DSV4/GLM shapes (the new
  `shall_use_sparse` threshold change does not regress them).
- `GGML_LOG_LEVEL=5 ./bin/test-backend-ops -o FLASH_ATTN_EXT -p "..."` emits
  the new CUDA-side line for each specialisation that wins, with all
  diagnostic fields populated. Example:

  ```
  sparse flash attention: device=0 cc=12.0 DKQ=256 DV=256 ncols1=1 ncols2=8 Q_tokens=1 K_len=4096 n_kv_max=512 sparse=true
  sparse flash attention: device=0 cc=12.0 DKQ=256 DV=256 ncols1=8 ncols2=8 Q_tokens=64 K_len=8192 n_kv_max=512 sparse=true
  ```
- `./bin/test-llama-archs`: qwen4exp regression passes at NMSE ~8.5e-08 on
  the RTX 5070 and 0 on the CPU.
- A pre-existing single GET_ROWS test (`n=256, m=5, r=4, be1=700, be2=100`)
  fails on the 11 810 MiB RTX 5070 with `cudaMalloc` OOM; it is unrelated to
  this task.
