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
`dsv4_compress_ratios[il]`. For typical configs (`indexer_top_k = 512`,
`r = 16`) this is 527.

### CUDA side

`ggml/src/ggml-cuda/fattn-common.cuh`, `ggml/src/ggml-cuda/fattn-mma-f16.cuh`,
`ggml/src/ggml-cuda/fattn.cu`:

- `ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse` now also accepts
  `DKQ == DV == 256` with `(ncols1, ncols2)` in `{(1, 8), (8, 8)}`. Without
  this the sparse FA kernel is never selected for qwen4's head shape.
- `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` takes `cc` and `ncols1`
  as explicit parameters. The `K->ne[1] >= ...` guard now scales with the full
  gather work for one tile: `2 * n_gather` where
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

`src/llama-graph.cpp` prints a one-shot `LLAMA_LOG_DEBUG` line the first time
`ggml_flash_attn_ext_set_n_kv_max` is called with a non-zero value. The line
is verbose-only, so it does not appear under the default log level. To verify
sparse FA is engaged during Maya benchmarking, set `-v 5` (or
`--log-level debug`) and look for the `sparse flash attention: n_kv_max=...`
message. The log is written once per process even when multiple contexts are
created.

## How the sparse bound is computed

For each attention layer with `r = dsv4_compress_ratios[il] > 0`, the indexer
picks the top `width = min(n_kv, indexer_top_k + r - 1)` cells per query
(whole `r`-sized blocks plus a possible tail). The mask build in
`build_attn_qsa` zeroes every other K position. The width is then passed as
`n_kv_max` to the flash attention op. The CUDA kernel stops its K iteration at
the live count of the sparse index list, so the per-tile work is bounded by
`width` instead of `n_kv`.

## Threshold and fallback behaviour

`ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` returns true only when:

- The hardware is NVIDIA Turing or later (the MMA kernel is the only path that
  implements the sparse gather).
- `DKQ == DV == 256` and `(ncols1, ncols2)` is `(1, 8)` or `(8, 8)`.
- The KQ mask has the expected shape, no ALiBi, no softcap, no sinks.
- `K->ne[1] >= max(4096, 2 * n_gather)`. With `ncols1 = 1` decode this is the
  per-sequence gather; with `ncols1 > 1` prefill this is per tile.

When any condition fails the dense/masked FA path runs. In particular, the
guard rejects qwen4 prefill at short-to-medium context where `n_gather` would
exceed half of `n_kv` and the gather is not worth it.

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

- `cmake --build . --target ggml-cuda llama test-backend-ops test-llama-archs`
  builds clean on the dual-3060 host with CUDA 13.4.
- `./bin/test-backend-ops -o FLASH_ATTN_EXT`: 3982/3982 tests pass, including
  the three new qwen4-shape sparse FA cases and the pre-existing sparse FA
  tests for DSV4/GLM shapes (the new `shall_use_sparse` threshold change does
  not regress them).
- `./bin/test-llama-archs`: qwen4exp regression passes at NMSE 8.88e-08 on
  CUDA and 0 on CPU (graph construction matches the reference).
- A pre-existing single GET_ROWS test (`n=256, m=5, r=4, be1=700, be2=100`)
  fails on the 12 GB RTX 3060 with `cudaMalloc` OOM, unrelated to this
  change.
