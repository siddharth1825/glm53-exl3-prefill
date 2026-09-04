# glm53-exl3-prefill

Prefill MoE kernel work for **GLM-5.3-Flash (EXL3 4 bpw trellis experts)** served by the MiaAI vLLM fork on
**two NVIDIA DGX Spark (GB10, sm_121)** with tensor-parallel 2.

Result (2026-09-03), same harness, cold prompts, medians of three:

| configuration | cold 32K tok/s | TTFT 32K | cold 8K | decode prose / code |
|---|---|---|---|---|
| stock image (2026-08-28), 2048-token chunks | 922 | 27.5 s | 862 | 25.3 / 55.1 |
| upstream PR77 fat-expert kernel, 7168-token chunks | 1146 | 22.1 s | 1101 | 24.1 / 49.6 |
| **+ grouped weight-stationary prefill (this repo)** | **1500** | **16.9 s** | **1493** | 24.6 / 47.0 |

Numerics: the new path is bit-identical to PR77's kernel at the GEMM level and agrees with the reference
expert loop at fp16 noise (rel L2 5.7e-4 on a full 288-expert layer). On the served model, true-token prompt
logprobs deviate from either upstream path by the same amount the two upstream paths deviate from each other.

## What the kernel does

`kernel/exl3_grouped_prefill.cu` runs the routed-expert MoE for a prefill chunk as one launch per stage over
all 288 experts, with a weight-stationary mainloop: each 16×16 trellis tile is decoded once per 128-row block
and reused across the block, K stages of 64 with a 2-stage cp.async pipeline, a swizzled activation tile, and
the H128 / sign-vector epilogue of the original kernel.

1. rows sorted by expert (GPU), per-expert M-block table built on the GPU (`build_blocks`)
2. gather + input Hadamard with the per-expert sign vector (`row_had`)
3. grouped gate|up GEMM straight from the stacked `[E, 2, ...]` trellis tensor (`grouped_gemm`, fp16 out)
4. GLM's clamped SwiGLU + down-projection Hadamard (`act_had`)
5. grouped down GEMM with fp32 atomic scatter by token and route weight (`grouped_gemm`, mode 1)

`kernel/exl3_fat_gemm_v2.cu` is the same mainloop as a drop-in for PR77's per-expert `exl3_fat_gemm`
(four shared-memory/pipeline variants; used for the bit-exactness and per-call timing tests).

`kernel/exl3_grouped_runtime.py` + `kernel/patch_exl3_grouped.py` hook the grouped path into vLLM's exl3
overlay at container start (`EXL3_GROUPED_PREFILL=1`); decode keeps the untouched graph-safe fused path.

Per-layer MoE time on a real layer, all 288 experts, skewed routing (ms, TP rank):

| tokens in chunk | overlay (fused + PR77 fat) | grouped |
|---|---|---|
| 1024 | 17.0 | 11.4 |
| 2048 | 25.9 | 13.7 |
| 4096 | 48.2 | 20.2 |
| 8192 | 99.3 | 33.2 |

## Layout

- `kernel/` CUDA sources, the JIT test harnesses (`test_fat_v2.py`, `test_grouped.py`), the serving hook,
  and the earlier tuned-fused-kernel patch generator output (+3% end to end, not adopted).
- `harness/` benchmark matrix, PR77 A/B, temp-0 and logprob gates, burst/concurrency tests, gateway log pull.
- `serving/gw-start.sh` the rank-aware boot script used by the gateway's custom-cluster launch.
- `bench/` results.json (all rows), logprob captures, burst logs, exported launch specs (secrets redacted).
- `docs/` BASELINE.md (running lab notes), KERNEL_PLAN.md, BENCHMARK.md, the design document
  (`trellis-prefill-design.html`), and the five research reports the design was built on.

## Building and testing

Everything compiles inside the MiaAI image rebuilt from their repo at 2026-09-02 main (PR77 included), which
ships the exllamav3 extension sources at `/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext`.

```
docker run --rm --gpus all -v /raid:/raid -v $PWD:/work -w /work --entrypoint python3 \
  glm53-flash-sm121:pr77 kernel/test_grouped.py --model /raid/models/GLM-5.3-Flash-EXL3-TR3-4bpw
```

Two sm_121 build gotchas the harnesses already handle: the CUDA 13.0.88 launch stub disagrees with the pip
wheel's older `crt/host_runtime.h` (only nvcc's own include dir may provide `crt/`), and multi-parameter
kernel templates trip the stub macro (the kernels take a single template parameter).

## Serving configuration that is live

`bench/specs_export.json` → `glm-exl3-grouped-7168-seqs16-nothr`: 7168-token chunks, right-sized sparse-indexer
workspace, gpu-memory-utilization 0.87, 16 sequences, graph capture to 128 tokens, 1M context, DFlash2 k=7.
The long-prefill threshold was tried and rejected (it caps a lone request to 2048 tokens per step and makes
equal-size bursts all finish last; see `docs/BASELINE.md`).

## Contributing

Issues and PRs welcome, especially on the kernel: the mainloop sits at ~75% of the GB10's bf16 peak on
large blocks and is latency-bound on thin experts; the open questions are listed in `docs/BASELINE.md`
and the design document. MIT licensed; see `NOTICE.md` for the upstream code this builds on.

## Open

- Code-path decode dropped from 55 to ~47 tok/s with the upstream image rebuild (present on the legacy path
  too, so not from these kernels). Not yet isolated.
- Next kernel steps: 64-row tiles for thin experts, a fresh step profile, then dense projections.
