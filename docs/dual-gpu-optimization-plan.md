# Dual RTX 3060 Optimization Plan

## Project goal

Optimize llama.cpp for this specific machine first, then add model-specific improvements on top.

Target hardware:

- 2 x NVIDIA GeForce RTX 3060 12 GB
- Compute capability: SM86
- ~23.9 GiB aggregate VRAM
- PCIe-connected, no NVLink
- Direct GPU-to-GPU P2P is working through NCCL when forced across PHB

Primary models:

1. Qwen3.8 27B / llama.cpp `qwen35` architecture
2. Qwen3.8-Flash-Next / llama.cpp `qwen4exp` architecture

The branch is **not** a Qwen4Exp-only fork. Generic dual-GPU improvements should benefit any compatible model. Model-specific code is layered on top.

## Branch policy

- `master` stays clean and tracks upstream `ggml-org/llama.cpp`.
- `dual-gpu-opt` is the integration branch.
- Each substantial optimization gets its own feature branch from the current `dual-gpu-opt` head.
- Do not combine several unmeasured optimizations in one commit.
- Every feature branch must be benchmarked before merge.
- GenerelSchwerz is reference code only. Do not merge its branch wholesale.

## Current measured baseline

Using the existing build `73a43d1f6 (10826)`, Qwen3.8 27B Q4_K Medium, tensor split, CUDA1/CUDA0, tensor split 6/5, Q8_0 KV, ubatch 1024:

### Tensor split with NCCL P2P forced across PHB

- pp4096: 795.82 +/- 2.93 t/s
- tg128: 30.02 +/- 0.06 t/s
- NCCL transport: `P2P/direct pointer`

### Tensor split with P2P disabled

- pp4096: 754.41 +/- 10.23 t/s
- tg128: 29.19 +/- 0.12 t/s
- NCCL transport: `SHM/direct`

Observed P2P gain on this workload:

- prompt processing: about +5.5%
- token generation: about +2.8%

These numbers are reference data only. They are not yet the baseline for the new fork because the new fork has not been built and measured on this machine.

## Architecture of the work

```text
dual RTX 3060 runtime
|
+-- generic dual-GPU work
|   +-- P2P / NCCL transport
|   +-- tensor-parallel correctness
|   +-- tensor placement
|   +-- per-GPU VRAM accounting
|   +-- workspace placement
|   +-- PCIe traffic reduction
|
+-- Qwen3.8 27B / qwen35
|   +-- tensor-parallel tuning
|   +-- recurrent-state placement/tuning if needed
|   +-- long-context tuning if measurements justify it
|
+-- Qwen3.8-Flash-Next / qwen4exp
    +-- QSA indexer scratch chunking
    +-- QSA decode optimization
    +-- incremental pooled-key cache
    +-- MoE expert cache
    +-- per-GPU expert-cache budgeting
```

## Exact order of operations

### Phase 0 - Freeze the current reference data

Do not change code yet.

Record the current machine and test parameters:

- GPU model and VRAM
- driver version
- CUDA toolkit version
- NCCL version
- motherboard / PCIe topology (`nvidia-smi topo -m`)
- CPU and RAM
- exact llama.cpp commit
- exact model path and quant
- exact benchmark command
- device order
- split mode
- tensor split
- batch / ubatch
- KV types

Save the information under:

```text
benchmarks/dual-3060/system-info.txt
```

Create benchmark directories:

```text
benchmarks/dual-3060/qwen27b/
benchmarks/dual-3060/qwen-flash-next/
```

### Phase 1 - Build untouched fork baseline

On the dual-3060 machine:

1. Clone `dcazrael/llama.cpp` into a new directory.
2. Checkout `dual-gpu-opt`.
3. Build for CUDA SM86.
4. Do not apply PRs or source modifications.
5. Verify both CUDA devices are detected.

Suggested build:

```bash
cmake -S . -B build-sm86 \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-sm86 -j"$(nproc)"
```

The result of this phase is the new reproducible baseline binary.

### Phase 2 - Characterize Qwen3.8 27B generic multi-GPU behavior

Use Qwen3.8 27B first because it fits in aggregate VRAM and does not add QSA or MoE expert streaming. This isolates the generic dual-GPU path.

Keep model, quant, KV types, batch, ubatch and GPU order fixed while changing only one variable at a time.

Run at minimum:

1. Layer split, P2P normal/default
2. Tensor split, P2P disabled
3. Tensor split, P2P forced across PHB

Then test tensor balance:

- 50/50
- 55/45
- 60/40
- current 6/5 ratio

Then test device order:

- CUDA0/CUDA1
- CUDA1/CUDA0

Record for every run:

- commit SHA
- model and quant
- context
- batch
- ubatch
- KV types
- split mode
- tensor split
- device order
- P2P mode
- pp4096 t/s
- tg128 t/s
- GPU0 peak VRAM
- GPU1 peak VRAM
- GPU utilization if collected
- transport chosen by NCCL
- success/failure

Do not optimize source until this matrix is complete.

### Phase 3 - Decide whether generic tensor-parallel work is justified

Compare layer split against tensor split.

Questions to answer:

- How much does tensor split improve PP?
- How much does tensor split improve TG?
- How much of that gain comes specifically from direct P2P?
- Is either GPU consistently the memory or compute bottleneck?
- Does device order materially change throughput?
- Does one tensor ratio clearly dominate?

Only after these are answered should generic runtime code be changed.

### Phase 4 - Profile generic dual-GPU traffic and placement

Investigate where PCIe traffic comes from in tensor mode.

Priority principle:

> Eliminating an unnecessary transfer is better than making that transfer faster.

Look for:

- CPU bounce paths that can remain GPU-local
- mirrored tensors that do not need to be mirrored
- reductions that should use direct GPU P2P
- recurrent/cache state placed on the wrong GPU
- scratch buffers allocated disproportionately on one GPU
- data repeatedly crossing GPUs because ownership is unclear

Any proposed change must identify:

1. the current transfer or allocation,
2. why it is unnecessary or poorly placed,
3. expected impact,
4. how the benchmark will prove or disprove the improvement.

### Phase 5 - Add per-GPU memory accounting

Before adding a Flash-Next expert cache, establish explicit per-GPU budgets.

Conceptually:

```text
GPU total VRAM
- model shard
- KV / recurrent state
- persistent buffers
- communication buffers
- worst-case compute workspace
- safety margin
= dynamic budget
```

Do this independently per GPU. Do not treat 24 GB aggregate VRAM as one homogeneous pool.

The goal is to make future dynamic allocations, especially expert caches, aware of the workspace each GPU must still reserve.

### Phase 6 - Re-benchmark Qwen3.8 27B

After generic runtime changes:

- rerun the exact Phase 2 matrix,
- compare against untouched baseline,
- reject changes that improve one metric by harming another disproportionately,
- retain only measured improvements.

Qwen3.8 27B is the control model for generic dual-GPU work.

### Phase 7 - Establish Flash-Next baseline on the same runtime

Only now move to Qwen3.8-Flash-Next.

Start without custom GenerelSchwerz features.

Measure:

- startup / graph reserve success
- 8k context
- 32k context
- 64k context
- 96k context
- 128k context if stable
- PP t/s
- TG t/s
- peak VRAM per GPU
- largest compute workspace if available

This tells us which Flash-Next-specific problems still exist on current upstream plus our generic dual-GPU changes.

### Phase 8 - QSA indexer scratch chunking

Reference implementation: PR #27752, especially commit `dba3221199c516b588c913cebaad4e3fec364bdc`.

Do not cherry-pick the GLM-specific commit directly unless source inspection proves it is cleanly reusable.

Instead:

1. study the GLM indexer chunking implementation,
2. locate the equivalent Qwen4Exp indexer path,
3. port only the token-chunking idea,
4. keep behavior unchanged at short contexts when possible,
5. benchmark PP, scratch VRAM and output correctness.

Target outcome:

- bounded indexer scratch,
- larger usable ubatch,
- no huge deep-context transient allocation,
- no correctness regression.

### Phase 9 - QSA decode optimization

Test competing approaches separately.

Candidate A:

- gather-based QSA decode (PR #28213 family)

Candidate B:

- CUDA sparse-FA Qwen path (PR #28770 family)

Do not combine them before individual A/B testing.

Benchmark on the actual RTX 3060 SM86 pair at multiple context depths.

Select based on measured PP/TG, VRAM and correctness on this hardware, not based on another user's A6000/Metal results.

### Phase 10 - Incremental pooled-key cache

Reference: PR #28699.

Important design point for this machine:

- pooled-key storage should remain local to the GPU that owns the relevant layers,
- do not create a single central pooled buffer that forces repeated cross-GPU traffic.

Benchmark deep-context TG and PCIe traffic before and after.

### Phase 11 - MoE expert cache

Only after generic memory accounting and QSA behavior are stable.

Requirements:

- host-backed expert pool/cache
- GPU-resident hot experts
- per-GPU cache capacity
- workspace-aware budgets
- safety margin
- instrumentation for hits, misses and host-to-device traffic

Budget rule:

```text
free VRAM
- worst-case required compute workspace
- safety margin
= maximum expert-cache budget
```

Study GenerelSchwerz for useful mechanisms such as:

- hit/miss tracking
- pinned host memory
- prefetch
- overlap
- logging

Do not port its live-context workspace implementation wholesale.

### Phase 12 - Multi-GPU Flash-Next expert placement

After the expert cache is correct on one logical path, optimize for both GPUs:

- expert cache budget independently per GPU
- layer-local expert residency where possible
- avoid routing hot expert data unnecessarily through one GPU
- avoid CPU staging when direct P2P can be used safely
- measure PCIe traffic, not just tokens/sec

## Benchmark discipline

Every optimization must be compared against its immediate parent commit with the same test parameters.

Minimum metrics:

```text
commit
model
quant
context
batch
ubatch
KV K type
KV V type
split mode
tensor split
device order
P2P mode
PP t/s
TG t/s
GPU0 peak VRAM
GPU1 peak VRAM
NCCL transport
success/failure
```

Do not claim an optimization from one run. Use repeated measurements when differences are small.

## Agent roles

### GPT / architecture-review role

Use for:

- deciding architecture
- reviewing upstream PRs
- comparing alternative implementations
- diagnosing non-obvious interactions
- final code review

### MiniMax / implementation role

Use for:

- larger contained implementation passes
- ports after the design is specified
- mechanical refactors with clear boundaries

### Local Qwen 3.8 27B role

Use as a repo-aware engineering assistant, not as project architect.

Good tasks:

- trace a call path
- locate ownership of a tensor
- compare two implementations
- explain allocator behavior
- analyze compiler/runtime logs
- implement a small approved patch
- review a specific diff

Avoid broad prompts such as `fix multi-GPU Qwen3.8`.

## Development workflow

Developer PC:

```text
OpenCode / agent
-> edit feature branch
-> local compile/static checks where useful
-> commit
-> push
```

Dual-3060 machine:

```text
git fetch
-> checkout/reset to exact commit
-> build SM86
-> run benchmark
-> save result with commit SHA
```

The benchmark machine should not contain independent uncommitted source changes.

## Immediate next actions

Do these next, in order:

1. Build `dual-gpu-opt` untouched on the dual-3060 machine.
2. Record complete system/topology information.
3. Run Qwen3.8 27B layer-split baseline.
4. Repeat tensor split with P2P disabled.
5. Repeat tensor split with P2P forced across PHB.
6. Test tensor ratios and device ordering.
7. Save results under `benchmarks/dual-3060/qwen27b/`.
8. Review the matrix before making any source-code optimization.

Do not start QSA or expert-cache work until the generic dual-GPU baseline is understood.
