# BASELINE — GLM-5.3-Flash EXL3 prefill optimization
Date: 2026-08-31. Read-only inventory of the serving stack; no changes made.

## Running stack (deployment 60, containers gw-glm-exl3-dflash-node{0,1})
- image: ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3  (FROM vllm/vllm-openai:glm53-flash-arm64-cu130)
- vllm 0.1.dev20051+g487ecf187 (day-0 main snapshot; glm5_next NOT in a release — upstream PR #53906)
- torch 2.13.0+cu130 · CUDA 13.0 · triton 3.7.1 · flashinfer 0.6.17
  (NOTE: 0.6.17 is the version tonyd2wild found NaN-prone on batch 64–256 for the
   NVFP4 MoE path; the EXL3 lane does not exercise that kernel. Relevant if lane B/C.)
- device cap (12,1) GB10 · TORCH_CUDA_ARCH_LIST=12.1a · FLASHINFER_CUDA_ARCH_LIST=12.1a
- overlay files bind-mounted from ~/glm53-exl3/overlay (repo @ b5ab809); boot script ~/glm53-exl3/gw-start.sh

## Serving flags (spec 11 glm-exl3-dflash — OUR deltas vs MiaAI production in [brackets])
- quantization exl3 · kv fp8 (fp8_ds_mla 656 B/tok/layer) · max-model-len 1,000,000
- GPU_MEM_UTIL 0.84 [MiaAI: 0.87] → KV pool 1,111,111 tokens / 11.66 GiB [theirs 1.67–1.75M]
- MAX_NUM_BATCHED_TOKENS 2048 (their P1 winner) · MAX_NUM_SEQS 4 · enforce-eager OFF
- SPEC_METHOD dflash D=7 draft-TP 2 [drafter prefill adds ~3% per their P8 note] · ABLIT=1
- GLM53_MIXED_PREFILL_CHUNK=skip · prefix caching ON · EXL3_FUSED_MOE=1 · TEMP_ROWS_FUSED=128 · ROW_TILE off

## MiaAI cold-prefill baseline (docs/improve-prefill.md, MNBT=2048 confirm ladder, THEIR rig, util 0.87)
8k 797 tok/s / 10.0 s · 12k 926 · 16k 958 · 100k 984 / 101.7 s · 256k 973 / 263 s · 300k 941 / 319 s
APC 8k follow-up: 7168/8004 hits, 1.28 s. Decode: structured 65.9 / prose 26.2 (no NaN).
OUR ladder must be re-measured on this box (util + DFlash deltas) before any A/B — protocol: docs/_run_cold_prefill.py, unique salt, flush cache, median of 3, warmed.

## Their profile of the 1.08 s / 1024-tok chunk (P0, applies to this image)
moe_forward_shared 623 ms (63%) = exl3_moe fused 134 + LinearEXL3 reconstruct 64 + index_add 52 + rest
aten::mm shared/dense/lm_head 102 · NCCL AR 71 · sparse MLA 43 · KDA/GDN ~63
CPU: aten::nonzero ~0.48 s/step (fat fallback) — but CUDA busy ≈ wall (1.086 vs 1.08 s).
Fat fallback fires on 91–98.6% of prefill MoE layers; max_rows = full chunk; ~10.5 fat experts/layer.

## Task A target — host-sync origins (overlay/exl3.py, read 2026-08-31)
1. apply_exl3_fused_moe L~612: counts.max().item()            — sync 1/step/layer (fat probe)
2. L~634: fat = (counts > cap).nonzero(); fat.tolist()        — sync 2 (fat id set to host)
3. apply_exl3_python_loop L328: torch.unique(ids).tolist()    — sync 3 (drives 288-iter python loop)
4. L342: (ids == raw).nonzero(as_tuple=True)                  — per FAT expert (~10.5×/layer ≈ 440/step)
   (only_experts filter runs BEFORE the nonzero, so thin experts skip it)
Semantics to verify before touching: the fused exl3_moe launch at L~630 runs with the FULL
expert_count while fat experts overflow cap=128 — the kernel must be skipping fat experts
(else the python-loop pass would double-add). Op-level test MUST assert no double count.

## Prior art already tried and REVERTED (do not repeat blind)
P2a row-tiling (−60% 8k) · P2b TEMP_ROWS=1024 single launch (−24% 100k) · MNBT 3584/4096 · MNBT 8192 forbidden (indexer smem).

## Op-level baseline (2026-09-02, image kernel, layer 3, TP rank 0/2, DeepSeek serving concurrently)
Correctness vs apply_exl3_python_loop: PASS, rel err ~9e-4 (fp16 noise); fat-expert double-count check PASS.
Timing per MoE layer (median of 5): bsz64 7.9 ms · 256 10.8 · 1024 16.5 · 2048 26.9 · 4096 59.4 (fat path engaged).
Throughput saturates ~600 token-slots/ms from 1K tokens up = the 16-row re-decode signature (linear in tokens).
Target for the M-tiled kernel: sub-linear growth from 1K→4K, and the fat path retired.

## M-tiled kernel, attempt 1 (2026-09-02): CORRECT but NOT FASTER
Patched exl3_gemm_inner (TILESIZE_M generalised) + M=64/N=128 MoE instance: correctness PASS on all
7 checks incl. skewed 715-row expert (rel ~9e-4). Timing vs same-binary forced M=16 (N=256):
bsz64 15.9 vs 7.4 ms · 256 19.0/9.2 · 1024 20.2/15.8 · 2048 26.0/26.8 · 4096 54.3/59.5 ·
skewed 1024 26.6/18.5 · 2048 33.1/29.2 · 4096 52.6/53.6. Only ~9% at 4K where passes drop 4x
=> B re-read/dequant is NOT the wall (<~10%). Confound: the M=64 instance ran N=128 (2x K-steps)
while the M=16 path runs N=256. Next: M in {32,64} at N=256, and ncu on the K-loop.

## ncu profile of the baseline exl3_moe_kernel (2026-09-02, bsz 2048 skewed, one launch = 16.3 ms)
Compute (SM) throughput 44.5% · Memory throughput 30% · L2 hit 77.7% — neither pipe saturated.
Occupancy 33.3% theoretical = achieved: ONE 512-thread block per SM (grid 48 = 6 groups x 8 SMs).
Limiters: 128 registers/thread (full RF) AND dynamic smem 92.16 KB/block — the launcher requests
SMEM_MAX for every block although this shape needs ~31 KB.
Issue: 0.44 warps issued/scheduler/cycle · No Eligible 55.5% of cycles · 1.0 eligible warp/scheduler
· 9.1 cycles per issued instruction. => STALL/LATENCY-BOUND: too few warps to hide the K-loop
latency (cp.async waits, ldsm, dequant chains, barriers). Attempt 2 = occupancy: smem sized to
need, __launch_bounds__(512, 2), concurrency 12 groups. M-tiling addressed the wrong limiter.

## Attempt 2 — occupancy (2026-09-02): NOT FASTER; ptxas explains it
ptxas -v: the baseline k4/N256/M16 instance is already at 128 regs WITH spills (84 B stores/188 B
loads); __launch_bounds__(512,2) instances spill 1.1 KB and run 1.8x SLOWER (2K: 48.8 vs 27.5 ms).
M64/N256 also loses (39.9 ms, spills). Only win: EXL3_MOE_SMEM_FIT=1 (request real smem, not
SMEM_MAX): ~5% across all sizes (2K uniform 26.0 vs 27.5; 1K 15.2 vs 16.2). Register-starved kernel.

## Attempt 3 — pipeline depth (2026-09-02): +10-14% uniform / +6% skewed, spills gone
FRAG_STAGES 3->2 cuts spills 84->16-20 B; SH_STAGES 8 best. k4/N256/M16 s8f2 + smem fit vs original:
1K 14.0 vs 16.2 · 2K 24.4 vs 27.5 · 4K 54.1 vs 59.8 · skewed 2K 27.6 vs 29.3 · 4K 51.0 vs 54.2 (ms/layer).
Env: EXL3_MOE_SMEM_FIT=1 EXL3_MOE_STAGES=s8f2 (EXL3_MOE_PREFILL_M=16). Plateau ~600-670 token-slots/ms
persists => structural ceiling (cross-SM group barriers per expert phase, lockstep per-K-step syncs).

## reconstruct kernel throughput (2026-09-02): already at DRAM bandwidth
8 experts x 3 matrices (101M weights) in 0.97 ms = 103M weights/ms; fp16 write 207 MB/ms + packed read
52 MB/ms = 259 GB/s (95% of 273). Full layer (288 experts, TP half) = 35 ms to fp16, ~18-20 ms to fp8.
Fused trellis kernel (best tuned) per layer: 1K 14 · 2K 24.4 · 4K 54 · 8K ~110 ms (linear in tokens).
=> dequant-once-to-fp8 + fp8 fused-MoE GEMM (fixed ~20 ms + ~7-15 ms GEMM) breaks even ~2.5K tokens,
   ~1.8x at 4K, ~3x at 8K. Needs: fp8 block-scaled dequant kernel (derive from reconstruct.cu),
   overlay dispatch by chunk size, MNBT 8192 (task D indexer smem), temp-0 + KLD gates.

## Tuned kernel (smem-fit + s8f2) END-TO-END, 2026-09-02, config 4b vs 4 (DFlash, same harness)
cold prefill 32K: 895/924/922 -> 926/947/949 tok/s (+3.0%)   cold 8K: 862/859/870 -> 883/891/885 (+2.6%)
warm 8K: 1872 -> 1910/1919 (+2.3%)   decode prose 26.3/25.3/24.0 -> 23.9/25.1/21.9 (-6%, noisy: DFlash
acceptance varies)   decode code 55.3/54.2/55.1 -> 56.6/55.5/51.7 (~0)
Micro-bench +10-14% on the MoE kernel = +3% at the layer/request level (MoE GEMM is ~1/4 of prefill
time; attention/KDA, dense layers, routing, TP allreduce untouched). Verdict: NOT worth shipping as
default (3% prefill, possible decode noise). Confirms dequant-once fp8 path is the only >=2x lever.
Temp-0: 9/12 identical content vs stock; control run (stock vs stock) pending -> see control.log.
Control (stock vs stock, same prompts, temp 0): 10/12 identical -> the engine itself is not
deterministic at temp 0 (DFlash spec decode + batching change reduction order). Tuned kernel 9/12 is
within that variance: temp-0 diff is NOT a usable numerics gate here. For the fp8 dequant path use a
logprob/KLD gate (top-k logprobs on fixed prompts, single request, stock vs candidate) instead.
Note: chat API returns reasoning under message["reasoning"], not "reasoning_content".

## Design 1 kernel (exl3_fat_gemm_v2, 2026-09-03) — first GPU result, worker, pr77 image
K64 variants (3-stage 60 KB / 2-stage 40 KB): BIT-IDENTICAL to PR77 exl3_fat_gemm (max abs diff 0), 4.9e-6 vs fp32 ref.
Per-expert call (ms), gate|up M x 4096 x 2048 then down M x 1024 x 4096, E2 -> v1(K64,2st):
  128 rows: 0.132/0.040 -> 0.093/0.032 · 256: 0.146/0.051 -> 0.093/0.050 · 512: 0.159/0.089 -> 0.159/0.075
  1024: 0.415/0.177 -> 0.239/0.146 · 2048: 0.710/0.427 -> 0.457/0.272  (=> 75 TFLOPS gate|up at 2048 rows, ~75% of bf16 peak)
K32 variants NaN (swizzle mask bug: chunk ^ (row&7) with 4 chunks/row) -> fixed to row & (chunks-1).
Small-M regime (<=256 rows): both kernels latency-bound (16 CTAs for N=2048 on 48 SMs). Next: ONE grouped
launch over all experts (sorted rows, per-expert offsets from a GPU prefix sum, w13_trellis[e] stride, no
gate|up concat copy, fused gather+suh-Hadamard), replacing the per-expert Python loop and the fused trellis
kernel for prefill. PR77 E2 also copies gate+up trellis into packed13 per fat expert per layer (~8 MB each).

## GROUPED single-launch prefill MoE (design 1, grouped form) — 2026-09-03, worker, pr77 image, layer 3, ALL 288 experts, skewed routing
Correctness vs apply_exl3_python_loop: rel L2 5.7e-4 (overlay current path: 6.0e-4 – 8.6e-4) => fp16 noise.
Per layer (ms): current overlay (fused exl3_moe + PR77 E2 fat) -> grouped:
  1K 16.98 -> 11.41 (1.49x) · 2K 25.87 -> 13.73 (1.88x) · 4K 48.24 -> 20.19 (2.39x) · 8K 99.31 -> 33.21 (2.99x)
Pipeline: sort rows -> build_blocks (GPU) -> row_had gather+suh H128 -> grouped_gemm gate|up (fp16) ->
act_had (clamp/silu, suh_down H128) -> grouped_gemm down with fp32 atomicAdd scatter. No host sync, no per-expert loop.
Next: serving integration (runtime patch, EXL3_GROUPED_PREFILL=1), bench row 4f, KLD gate.

## END-TO-END (2026-09-03 01:24) — bench rows 4c/4d/4f (same harness as the 09-01 matrix)
row 4c PR77 @2048:            cold32K 1047 (TTFT 24.2s) · cold8K 961  · warm8K 2059 · decode 24.1 / 46.3
row 4d PR77 @7168 (+rightsize, GMU 0.87): cold32K 1146 (22.1s) · cold8K 1101 · warm8K 1642 · decode 24.1 / 49.6
row 4f GROUPED @7168 (design 1 in the engine): cold32K **1500** (TTFT **16.9s**) · cold8K **1493** · warm8K 2148 · decode 24.6 / 47.0
=> +63% cold prefill vs the 09-01 stock row (922), +31% vs PR77 at the same chunk; decode unchanged by the grouped path.
Code-decode 55 -> 46-50 appears in every row on the new image (4c/4d/4f) -> image-level, see row 4e (legacy control).

## NUMERICS GATE (2026-09-03 01:40) — true-token prompt logprobs, 3 fixed texts, served engine, prefill path
legacy(reconstruct+hgemm) vs PR77(E2): prose mean|d| 0.171 p99 1.16 PPL +1.72% · code 0.0041 · numbers 0.0265
legacy vs GROUPED:                    prose mean|d| 0.183 p99 1.34 PPL -0.77% · code 0.0029 · numbers 0.0311
PR77   vs GROUPED:                    prose 0.194 p99 1.38 · code 0.0034 · numbers 0.0321
=> grouped path is inside the spread between the two accepted upstream paths on every text. PASS.
Row 4e (legacy path, new image, MNBT 2048): cold32K 906 · cold8K 856 · decode 23.1 / 46.8 -> code-decode 55->47 is IMAGE-level
(present with legacy too), not from PR77/grouped kernels. Open item.

## Concurrency experiment (2026-09-03 17:00) — burst_test.py prompts are ~70.5K tokens (digit-heavy text; the "30000" arg under-estimates)
seqs4 (spec 17, no threshold):         single cold 70.5K TTFT 44.3s (1,590 tok/s) · 4-burst TTFT 48 / 93 / 138 / 178 s (serial FCFS at 7168 chunks)
seqs16 + long-prefill-threshold 2048:  single cold 70.5K TTFT 53.3s (1,323 tok/s, -17%: the threshold caps ONE request to 2048 tok/step even when alone)
                                       4-burst TTFT 182 / 187 / 187 / 212 s — round-robin makes everyone finish last, and 2048-token slices lose the
                                       sublinear MoE gain. Verdict: the threshold is wrong for equal-size bursts; only helps short-behind-long.
KV pool at seqs16: 1,357,142 tokens (20.87 GiB; graph memory estimated 0 with the 128 capture ceiling).
Next: spec 19 = seqs16 WITHOUT the threshold (FCFS at full chunk efficiency + 16 decode slots).
