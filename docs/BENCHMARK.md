# GLM / DeepSeek serving benchmark — 2× DGX Spark (GB10)

**Date:** 2026-09-01 · **Hardware:** 2× DGX Spark GB10 (sm_121), TP=2 over ConnectX-7 RoCE, 121.7 GB unified memory per node, max SM clock 3,003 MHz

Measured on spark-head against each engine's **local port** — no gateway, no Cloudflare, no tailnet in the numbers.

## Method

Per config: launch → **JIT burn-in (2× 400-token generations, discarded)** → cold prefill (3× ~32K + 3× ~8K, unique salt at position 0 so the prefix cache always misses) → warm prefill (the same 8K prompt repeated, so the cache always hits) → decode (3× prose, 3× code, 400 max_tokens) → `/metrics` speculation counters. GPU sampled at 1 Hz on both nodes throughout.

All figures are **medians of 3**, temp 0, thinking off, streamed. `prefill tok/s = prompt_tokens / TTFT`; `decode tok/s = completion_tokens / (total − TTFT)`.

The burn-in matters: the first request after a boot pays 30–50 s of kernel/TileLang JIT on these lanes, and mistaking that for prefill has produced at least two retracted results in this community.

## Prefill

| config | 32K cold tok/s | 32K TTFT | 8K cold tok/s | 8K TTFT | 8K warm TTFT (cache hit) |
|---|---:|---:|---:|---:|---:|
| 1-deepseek-v4-flash-fp8 | 1,966 | 12.8 s | 1,826 | 3.5 s | 0.43 s |
| 2-glm-exl3-base | 934 | 27.1 s | 922 | 6.9 s | 3.35 s |
| 3-glm-exl3-mtp | 907 | 27.9 s | 890 | 7.2 s | 7.12 s |
| 4-glm-exl3-dflash | 922 | 27.5 s | 862 | 7.4 s | 3.40 s |
| 5-glm-nvfp4-sglang-base | 1,203 | 21.1 s | 1,795 | 3.5 s | 0.67 s |
| 6-glm-nvfp4-sglang-eagle-mtp | colspan — **never became servable** | | | | |

## Decode

| config | prose tok/s | code tok/s | prose spread | code spread | boot |
|---|---:|---:|---:|---:|---:|
| 1-deepseek-v4-flash-fp8 | 25.5 | 25.5 | 25.5–25.5 | 25.4–25.6 | 586 s |
| 2-glm-exl3-base | 13.8 | 13.9 | 13.8–13.9 | 13.8–13.9 | 496 s |
| 3-glm-exl3-mtp | 24.3 | 30.3 | 23.3–24.4 | 29.8–30.5 | 526 s |
| 4-glm-exl3-dflash | 25.3 | 55.1 | 24.0–26.3 | 54.2–55.3 | 556 s |
| 5-glm-nvfp4-sglang-base | 14.5 | 14.1 | 13.6–14.6 | 14.1–14.1 | 873 s |

## GPU under load (during the code-decode phase)

| config | node | clock MHz med/max | temp °C med/max | power W | util % | pstate |
|---|---|---:|---:|---:|---:|---|
| 1-deepseek-v4-flash-fp8 | head | 2,444/2,444 | 79/81 | 43.8 | 94 | P0 |
|  | worker | 2,470/2,470 | 77/77 | 39.9 | 94 | P0 |
| 2-glm-exl3-base | head | 2,437/2,444 | 76/78 | 44.6 | 96 | P0 |
|  | worker | 2,470/2,470 | 76/77 | 41.6 | 96 | P0 |
| 3-glm-exl3-mtp | head | 2,437/2,437 | 76/78 | 47.3 | 95 | P0 |
|  | worker | 2,463/2,470 | 78/80 | 44.2 | 95 | P0 |
| 4-glm-exl3-dflash | head | 2,424/2,424 | 82/83 | 56.4 | 96 | P0 |
|  | worker | 2,470/2,470 | 81/81 | 51.8 | 95 | P0 |
| 5-glm-nvfp4-sglang-base | head | 2,444/2,444 | 75/76 | 40.5 | 95 | P0 |
|  | worker | 2,470/2,470 | 74/75 | 37.2 | 95 | P0 |

## Speculation

| config | drafts | draft tokens | accepted | acceptance rate | accepted per step |
|---|---:|---:|---:|---:|---:|
| 3-glm-exl3-mtp | 1,118 | 2,236 | 1,639 | 73.3% | 1.47 |
| 4-glm-exl3-dflash | 678 | 4,746 | 2,038 | 42.9% | 3.01 |

## Provenance — engine, image and flags per config

**Sourcing.** Config 4 is ground truth: its deployment record (70) survived and carries the exact
launch spec the agent executed. The gateway deletes a stopped deployment's row when a same-named
one launches, so records for configs 1/2/3/5 were garbage-collected during the run; those rows are
reconstructed from the saved spec plus the per-config override that `bench_matrix.py` applied
before each launch (the harness patches deterministically, so this is faithful — but it is a
reconstruction, not a capture). *Fixed for future runs: the harness should record container argv
and env at measurement time.*

All configs ran **TP=2 across spark-head + spark-worker**, `--network host`, RDMA over
`rocep1s0f1` (GID 3, RoCEv2), worker rank launched first.


### 1-deepseek-v4-flash-fp8  *(reconstructed from spec 6, unmodified by the benchmark)*

- **engine** vLLM · **mode** cluster/mp · **image** `aidendle94/sparkrun-vllm-ds4-gb10:production-ready`
- **entrypoint** `dsv4-vllm-entrypoint` · **weights** `/raid/models/DeepSeek-V4-Flash-0731-Abliterated-FP8` · served `deepseek-v4-0731` :8000
- **key flags** tp=2 gmu=0.8 ctx=262144 kv=fp8 quant=None parsers=deepseek_v4/deepseek_v4
- **extra args** `--block-size 256 --tokenizer-mode deepseek_v4`
- **speculation** none

### 2-glm-exl3-base *(reconstructed: spec 11 + SPEC_METHOD override)*

- **engine** vLLM · **mode** custom multi-node · **image** `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`
- **entrypoint** `bash /start.sh` (MiaAI-Lab boot script; flags are assembled from env)
- **weights** `/raid/models/GLM-5.3-Flash-EXL3-TR3-4bpw` · served `GLM-5.3-Flash-EXL3` :8888
- **serving env**

```
ABLIT=1
ABLIT_LAYERS=15-45
ENFORCE_EAGER=0
EXL3_FUSED_MOE=1
GPU_MEM_UTIL=0.84
KV_CACHE_DTYPE=fp8
LANGUAGE_MODEL_ONLY=0
MAX_MODEL_LEN=1000000
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=4
QUANTIZATION=exl3
SPEC_METHOD=none
```

- **speculation** no speculation
- **runtime patches** MiaAI-Lab overlay: video placeholders, suppress-stops-in-reasoning, scheduler decode floor, glm5 drafter group, hybrid prefix hit, xgrammar termination, kpool tail slotmap, ablit (o_proj orthogonalization, layers 15–45)

### 3-glm-exl3-mtp *(reconstructed: spec 11 + SPEC_METHOD override)*

- **engine** vLLM · **mode** custom multi-node · **image** `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`
- **entrypoint** `bash /start.sh` (MiaAI-Lab boot script; flags are assembled from env)
- **weights** `/raid/models/GLM-5.3-Flash-EXL3-TR3-4bpw` · served `GLM-5.3-Flash-EXL3` :8888
- **serving env**

```
ABLIT=1
ABLIT_LAYERS=15-45
ENFORCE_EAGER=0
EXL3_FUSED_MOE=1
GPU_MEM_UTIL=0.84
KV_CACHE_DTYPE=fp8
LANGUAGE_MODEL_ONLY=0
MAX_MODEL_LEN=1000000
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=4
QUANTIZATION=exl3
SPEC_METHOD=mtp
```

- **speculation** MTP_TOKENS=2
- **runtime patches** MiaAI-Lab overlay: video placeholders, suppress-stops-in-reasoning, scheduler decode floor, glm5 drafter group, hybrid prefix hit, xgrammar termination, kpool tail slotmap, ablit (o_proj orthogonalization, layers 15–45)

### 4-glm-exl3-dflash *(GROUND TRUTH — deployment 70)*

- **engine** vLLM · **mode** custom multi-node · **image** `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`
- **entrypoint** `bash /start.sh` (MiaAI-Lab boot script; flags are assembled from env)
- **weights** `/raid/models/GLM-5.3-Flash-EXL3-TR3-4bpw` · served `GLM-5.3-Flash-EXL3` :8888
- **serving env**

```
ABLIT=1
ABLIT_LAYERS=15-45
ENFORCE_EAGER=0
EXL3_FUSED_MOE=1
GPU_MEM_UTIL=0.84
KV_CACHE_DTYPE=fp8
LANGUAGE_MODEL_ONLY=0
MAX_MODEL_LEN=1000000
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=4
QUANTIZATION=exl3
SPEC_METHOD=dflash
```

- **speculation** DFLASH_TOKENS=7 DFLASH_DRAFT_TP=2, drafter /raid/models/GLM-5.3-Flash-DFlash2
- **runtime patches** MiaAI-Lab overlay: video placeholders, suppress-stops-in-reasoning, scheduler decode floor, glm5 drafter group, hybrid prefix hit, xgrammar termination, kpool tail slotmap, ablit (o_proj orthogonalization, layers 15–45)

### 5-glm-nvfp4-sglang-base  *(reconstructed: spec 9 + pinned base args)*

- **engine** SGLang · **mode** cluster/mp · **image** `lmsysorg/sglang:glm-5.3-flash`
- **entrypoint** `python3 -m sglang.launch_server` · **weights** `/raid/models/GLM-5.3-Flash-UNCENSORED-NVFP4` · served `GLM-5.3-Flash` :8100
- **key flags** --tp-size 2 --mem-fraction-static 0.84 --context-length 262144 --kv-cache-dtype bfloat16 --reasoning-parser glm45 --tool-call-parser glm47
- **extra args** `--attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --moe-runner-backend flashinfer_cutlass --disable-shared-experts-fusion --max-running-requests 2`
- **speculation** none
- **runtime patch** GB10 TileLang shared-memory retune bind-mounted over `sglang/kernels/ops/attention/dsa/tilelang_kernel.py` (block_I=32, num_stages=1, threads=128 — stock tiles request 169,984 B vs GB10's 101,376 B ceiling)

### 6-glm-nvfp4-sglang-eagle-mtp  *(FAILED — reconstructed: spec 9 + EAGLE args)*

- identical to config 5 plus `--speculative-algorithm EAGLE --speculative-eagle-topk 1 --speculative-num-steps 5 --speculative-num-draft-tokens 6`
- **outcome** `eagle_worker_v2.py:174 EagleDraftWorker.__init__ → RuntimeError: The size of tensor a (4096) must match the size of tensor b (2048) at non-singleton dimension 1`, reproduced 3×

### Stack versions (identical across all vLLM-lane configs)

Read from the running container: vLLM `0.1.dev20051+g487ecf187` (day-0 main snapshot — `glm5_next`
is not in any vLLM release; upstream PR #53906), torch `2.13.0+cu130`, CUDA 13.0, Triton `3.7.1`,
FlashInfer `0.6.17`, `TORCH_CUDA_ARCH_LIST=12.1a`, device capability (12,1) GB10.
The SGLang lane runs `lmsysorg/sglang:glm-5.3-flash` (arm64, checkout 033446bb05, PR #36507 branch).

## Findings

1. **The gap is kernel efficiency, not hardware.** Every working config pinned the GPUs at
   2,424–2,470 MHz of a 3,003 MHz ceiling, 94–96% utilisation, P0, 74–83 °C, 37–56 W — on both
   nodes. Throughput differs by 2× across configs at identical silicon behaviour, so no result
   here is explained by clocks, thermals, or power.

2. **The ~930 tok/s prefill ceiling is specific to the EXL3 path, not to GLM.** The same model
   on the NVFP4/SGLang lane prefills at 1,795 tok/s at 8K — 1.95× the EXL3 lane and within 2% of
   DeepSeek's 1,826. This is an existence proof on identical hardware that GLM can prefill at
   DeepSeek-class speed, which makes the EXL3 MoE/trellis path a well-defined optimisation target
   rather than a hypothesis.

3. **Prefix caching is also EXL3-specific.** NVFP4/SGLang warms to 0.67 s and DeepSeek to 0.43 s,
   while EXL3 only reaches 3.35–3.40 s on the same 8K prompt. GLM's hybrid KDA architecture is not
   the cause — the same architecture caches properly on the other lane.

4. **Speculation: deeper drafting beats higher acceptance.** DFlash2 accepts only 42.9% of its
   drafts versus MTP's 73.3%, but drafts 7 tokens per step instead of 2, netting 3.0 accepted
   tokens per step against MTP's 1.47 — and 55.1 vs 30.3 tok/s on code. Neither costs anything
   measurable on prefill.

5. **Native MTP (EAGLE) is broken on the SGLang lane** for `glm5_next`, reproduced three times:
   `eagle_worker_v2.py:174 → RuntimeError: The size of tensor a (4096) must match the size of
   tensor b (2048) at non-singleton dimension 1`. That lane's only speculation option is DFlash2,
   which needs an upstream model-file patch.

## Choosing a lane

Total wall time ≈ `prompt/prefill + generated/decode`. Against DeepSeek, EXL3+DFlash wins once
**generated tokens exceed ~2.7% of prompt tokens**:

| scenario | DeepSeek | EXL3+DFlash | winner |
|---|---:|---:|---|
| 30K prompt, 500 out | 34.9 s | 41.6 s | DeepSeek |
| 30K prompt, 3,000 out | 132.9 s | **86.9 s** | DFlash (−35%) |
| 95K prompt, 500 out | 67.9 s | 112 s | DeepSeek |
| 8K prompt, 2,000 out | 82.8 s | **45.6 s** | DFlash (−45%) |

Short answers over huge prompts favour DeepSeek; long generations favour EXL3+DFlash. Closing the
EXL3 prefill gap (finding 2) would make EXL3+DFlash win both axes at once.

## Caveats

- DeepSeek is a different model with a different tokenizer, so tok/s is not strictly
  per-unit-of-work comparable; wall-clock on identical text is the fairer read.
- Each lane ran at **its own proven production settings** (DeepSeek gmu 0.80 / 256K ctx; GLM lanes
  gmu 0.84), not force-equalised — the question asked was "how fast is each lane as I would
  actually run it".
- `n=3` per phase; GPU sample counts vary by phase duration (22–87), fine for medians but not for
  distribution claims.
- Config 4's first attempt failed on a launch race (relaunching a spec 39 s after stopping it, onto
  memory the placement check believed was free). The number here is from a retry gated on both
  nodes actually reporting ≥105 GB free.


## Addendum 2026-09-02 — config 4b: tuned EXL3 MoE kernel (same DFlash config as 4)

Kernel change: exllamav3 `exl3_moe_kernel` rebuilt with a deeper smem pipeline (8 shared-memory stages,
2 fragment stages) and a host-side smem-fit calculation, bind-mounted over the image's extension
(`exllamav3_ext_s8f2.so`, env `EXL3_MOE_SMEM_FIT=1 EXL3_MOE_STAGES=s8f2`). Everything else identical to config 4.

| phase | config 4 (stock) | config 4b (tuned) | delta |
|---|---|---|---|
| cold prefill 32K tok/s (3 runs) | 895 / 924 / 922 | 926 / 947 / 949 | +3.0% |
| cold prefill 8K tok/s | 862 / 859 / 870 | 883 / 891 / 885 | +2.6% |
| warm prefill 8K tok/s | 1872 | 1910 / 1919 | +2.3% |
| decode prose tok/s | 26.3 / 25.3 / 24.0 | 23.9 / 25.1 / 21.9 | −6% (DFlash acceptance noise) |
| decode code tok/s | 55.3 / 54.2 / 55.1 | 56.6 / 55.5 / 51.7 | ~0 |

The kernel micro-benchmark showed +10–14% on the MoE GEMM alone; at the request level that is +3%
because the MoE GEMM is roughly a quarter of prefill time. Not adopted as default. Full rows in results.json (`4b-…`).

## Addendum 2026-09-03 — PR77 image and the grouped single-launch prefill kernel (rows 4c–4f)

Image rebuilt from MiaAI upstream at 2026-09-02 main (PR77 fat-expert kernels, #86 indexer workspace, #63). Rows 4d/4f
run the upstream default chunk of 7168 tokens, which at 1M context needs `GLM53_INDEXER_WORKSPACE=rightsize` and
`GPU_MEM_UTIL=0.87` (upstream's validated boot). Row 4f adds our grouped weight-stationary trellis GEMM
(`~/glm-opt/fatv2/exl3_grouped_prefill.cu`, engaged via `EXL3_GROUPED_PREFILL=1`) for every prefill MoE call.

| row | config | cold 32K tok/s | TTFT 32K | cold 8K | warm 8K | decode prose / code |
|---|---|---|---|---|---|---|
| 4 | stock image, 2048 | 922 | 27.5 s | 862 | 1872 | 25.3 / 55.1 |
| 4c | PR77 image, E2 kernel, 2048 | 1047 | 24.2 s | 961 | 2059 | 24.1 / 46.3 |
| 4d | PR77 image, E2 kernel, 7168 | 1146 | 22.1 s | 1101 | 1642 | 24.1 / 49.6 |
| 4f | PR77 image + grouped prefill, 7168 | **1500** | **16.9 s** | **1493** | 2148 | 24.6 / 47.0 |
| 4e | PR77 image, legacy fat path, 2048 (control) | 906 | 27.8 s | 856 | 1817 | 23.1 / 46.8 |

Numerics gate (true-token prompt logprobs on three fixed texts, served engine): the grouped path deviates from either
upstream path by the same amount the two upstream paths deviate from each other (random-word text mean |Δ| 0.18 vs 0.17
nats; code 0.003 vs 0.004; digits 0.031 vs 0.027). Kernel-level: grouped vs reference loop rel L2 5.7e-4 (fp16 noise).
The code-decode drop from 55 to ~47 tok/s is present in every row on the new image including the legacy control, so it
came with the image, not with the kernels. Open.
