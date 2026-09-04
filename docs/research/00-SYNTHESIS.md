# GLM-5.3-Flash prefill on 2× DGX Spark — research synthesis and design (2026-09-03)

Five sourced reports sit next to this file (01–05). This document is the synthesis: what the
research corrected in our mental model, what the numbers now say, and a ranked design. Everything
here was cross-checked against the running box on 2026-09-03 (image built 2026-08-28).

## 1. What we had wrong

| Belief we worked from | What the research established | Source |
|---|---|---|
| GLM-5.3-Flash is a "30B-A3B-class" model | **320B total / 18B active.** 288 experts × 25.2M params × 42 MoE layers ≈ 304B routed params. At 4 bpw each Spark holds ~76 GB of expert weights and streams all of them every prefill chunk. | 01, 02, config.json |
| Prefill is bandwidth-bound; a 3× faster MoE means ~1.4× prefill | **Prefill is kernel-efficiency-bound, not bandwidth-bound.** Weight floor 0.31 s/chunk; compute floor ~1.45 s per 8K chunk at bf16; we measure ~9 s. The MoE runs at ~13–15% of the bf16 tensor peak. | 01, 02, 04 |
| Dequant-once to fp8/bf16 scratch is the way to a 2×+ MoE | **Wrong on this hardware.** Writing dequantized experts adds 150–300 GB/step of traffic. The roofline-correct design is weight-stationary: decode each trellis tile once into registers, sweep 64–128 activation rows over it. exllamav3's own dense GEMM already does dequant-once above 144 rows; nobody has shipped the weight-stationary trellis GEMM. | 01, 04 |
| fp8 w8a8 is "official FP8 class" numerics | It is only if the activations are quantized in the **Hadamard-rotated domain with per-token 128-groups aligned to the H128 blocks**. Our prototype quantized raw activations → 6.3%. Rotated **int8** W8A8 is ≈1–1.5% error (QuaRot INT8: ppl 5.50 vs 5.47 fp16) and runs at the fp8 tensor rate on sm_121. | 04 |
| DeepSeek-V4-Flash is faster because it is "FP8" | V4-Flash's experts are **native MXFP4 (QAT)**, rest FP8. Its 2× prefill decomposes as ≈1.3× fewer active FLOPs (13B vs 18B, top-6 vs top-8) × ≈2× datapath ceiling (fp8/fp4 tensor cores at ~208 TFLOPS vs fp16 at ~100) — both lanes sit at ~15–18% of their respective ceilings. Expert granularity is identical. | 03 |
| Our 900 tok/s is behind the community | It **is** the community number for vLLM-EXL3 pre-PR77 (938/997/941). SGLang-NVFP4 ~1,800 at 8K is also state of the art. Nobody has a fundamentally better kernel for this box yet. | 02, 05 |
| MNBT 8192 is forbidden (indexer smem) | MiaAI patched the indexer workspace (PR #86) and their **current default is MNBT 7168** with the PR77 fat-expert kernel. | 05, upstream README |

## 2. The numbers that matter (per TP rank, GB10)

- GB10 measured: bf16 99.8 TFLOPS, fp8 207.7 TFLOPS, int8 = fp8 rate, 273 GB/s (232 sustained), **99 KB smem per block**, `mma.sync` only (no WGMMA/TMEM/tcgen05), no practical native FP4 path yet except FlashInfer's `b12x`.
- Per MoE layer per rank: 3.62 G expert params = 1.81 GB at 4 bpw (6.6 ms DRAM), 3.62 GB int8 (13.3 ms), 7.2 GB fp16 (26.5 ms). FLOPs = tokens × 201 MFLOP → 0.41 / 0.82 / 1.65 TFLOP at 2K / 4K / 8K tokens.
- Current fused trellis kernel: 14 / 24 / 54 / 110 ms per layer at 1K / 2K / 4K / 8K = ≈15 TFLOPS at every size. It processes exactly 16 activation rows per pass and re-decodes every weight tile for every pass (`static_assert(TILESIZE_M == 16)`).
- Step anatomy (image authors' P0 profile, MNBT 1024): MoE forward 63%, dense GEMMs 10%, NCCL allreduce 7%, KDA 6%, sparse MLA 4%. Our own 2K measurement: MoE kernel ≈45% of the 2.28 s step, the rest is glue/attention/allreduce.
- mHC: fused TileLang kernels confirmed active on our box (`mhc_pre_big_fuse_with_norm_tilelang` JIT in the log), so it is not the hidden cost the architecture report feared.

## 3. Ranked designs for the MoE prefill kernel

Per-layer MoE time, per rank. "Exact" = bit-comparable to today's path (same codebook, same rotations).

| # | Design | 2K | 8K | Numerics | Effort | Confidence |
|---|---|---|---|---|---|---|
| 0 | **Pull MiaAI PR77 image** (E2 fat-expert kernels: sort rows by expert, batched fat-expert GEMM) + MNBT 7168 | +20% end-to-end prefill (941→1,132 @8K, 1,023→1,242 @100K) | | exact | none (redeploy) | high (measured upstream) |
| 1 | **Weight-stationary trellis GEMM**: stage one expert's A rows (64–128) in smem via cp.async, decode each 16×16 B tile once into `m16n8k16` fragments and reuse across the M pass, per-expert grid with ticket scheduler | ≈10–13 ms (vs 24) | ≈25–35 ms (vs 110) | exact | medium: mainloop rewrite of `exl3_gemm_inner.cuh` / `exl3_moe_kernel.cuh` | high on direction, medium on magnitude |
| 2 | **Rotated-domain int8 W8A8**: decode W′ to int8 (per-row scale; W′ is Gaussian in the rotated domain), fuse gather + su-sign + H128 + per-token-128-group int8 quant of x, `m16n8k32.s8` grouped GEMM with int32→fp32 promotion per 128-k, H128 + sv in epilogue | ≈23 ms (break-even) | ≈31 ms | ≈1–1.5% (needs KLD gate) | medium-high (3 kernels) | high on numerics, medium on speed |
| 3 | Design 1 with int8 MMA (decode trellis straight to int8 fragments, int8 A in the gather) | ≈8–10 ms | ≈20–25 ms | ≈1% | high | medium |
| 4 | Fix our fp8 Triton path (rotated-domain quant, tuned config within 99 KB) | ≈40 ms | ≈45 ms | ≈4% | low | stopgap only |
| 5 | NVFP4 W4A4 via FlashInfer b12x / CUTLASS SM120 grouped | needs re-quantizing the trellis weights (double quantization) + 4-bit activations (+0.5 ppl class) | | worse than EXL3 | medium | not recommended |
| 6 | Convert experts to GPTQ int4 + Marlin MoE | ≈fp16 GEMM speed to M=1024 | | loses trellis quality (KLD 0.025 → ~0.05 class) | medium | not recommended |

End-to-end implication (MoE ≈ 63% of the step): design 1 alone gives ≈1.45× prefill at 2K chunks
and ≈1.8× at 8K chunks; with PR77's row sorting and MNBT 7168 on top, 900 → roughly 1,600–1,900 tok/s
cold, at exact numerics. That is parity with the SGLang NVFP4 lane while keeping the EXL3 4 bpw
quality (KLD 0.025 vs 0.04–0.06 for NVFP4).

## 4. Why my previous attempts did not move the needle

- **M-tiling (64 rows per pass in the same kernel)**: correct but not faster because it kept the existing structure (B decoded per pass into registers already at 128 regs with spills, 33% occupancy, 31% barrier stalls). Design 1 is not "bigger M in the same loop"; it is Marlin's loop: A batch tile stationary in shared memory, decoded B fragments held across the pass, deep cp.async pipeline. The register budget works: 128×128 fp32 accumulators = 64 regs/thread over 256 threads; two 128-row × 128-k fp16 A stages = 64 KB < 99 KB.
- **Occupancy / pipeline-depth variants**: +10–14% on the kernel, +3% end-to-end. Confirms the kernel is stall-bound, not decode-throughput-bound, and only a mainloop redesign changes that.
- **fp8 dequant-once**: 6.3% error because activations were quantized un-rotated, and the scratch traffic makes it a net loss below ~4K tokens anyway.

## 5. Recommended plan (kernel scope)

1. **Now, no code**: redeploy from the MiaAI image with PR77 (`EXL3_FAT_KERNEL=1`, upstream default) and MNBT 7168. Re-run our benchmark harness (configs 4/4b) so the +20% and the larger chunk are measured on our box, not assumed. Note the image also carries the reasoning-effort prefix-cache fix (#63) and the indexer workspace right-sizing (#86).
2. **Build design 1** on top of that image's exllamav3 (the fat-expert path is where prefill rows go; PR77's E2 kernels are the natural home for a weight-stationary mainloop). Gate: bit-for-bit or fp16-noise equality with the reference loop in the op harness; then the KLD/logprob gate on the served model (temp-0 diffs are not usable: the stock engine is only 10/12 reproducible).
3. **Then decide on int8** (design 2/3) only if design 1 lands short of ~25 ms at 8K and the KLD gate budget allows ~1% activation error.
4. **Do not spend time on**: fp8 w8a8 (numerics), NVFP4 re-quant (quality), Marlin conversion (quality), dequant-to-HBM (bandwidth).

## 6. Findings outside the kernel scope (parked, listed for completeness)

These came out of the research and each addresses the original TTFT complaint without touching model capability. Not acted on, per the kernel-only instruction.

- `--long-prefill-token-threshold 1024`: MiaAI issue #110 shows concurrent sessions freezing 440 s behind one long prefill → 18–25 s with this flag, ~10% cost to the long prefill. vLLM V1 has no other partial-prefill control.
- `MAX_NUM_SEQS=4` is the recipe's choice, not an engine limit; 8–12 is safe (>12 hits MTP profiling bug #88). KV pool is block-allocated; a 1M context does not reserve 1M tokens per sequence.
- Cloudflare 524: SSE heartbeat comments during prefill reset the 100 s timer; or grey-cloud the host.
- DFlash2 goes negative beyond ~6 concurrent streams on 2×GB10; `num_speculative_tokens_per_batch_size` tapers k with batch size. The drafter (`incoai/GLM-5.3-Flash-DFlash2`) is CC BY-NC-ND.
- Spec-decode drafter state fragments the hybrid KDA+MLA cache and can halve long-prefill speed after mixed workloads (NVIDIA forum thread); `--mixed-prefill-token-cap -1` helped there.
- EP=2 instead of TP=2 for the experts (full 2048-wide expert shapes, one all-to-all instead of two allreduces per layer) is what the 2× RTX PRO 6000 EXL3 recipe uses at 4,200–6,200 tok/s. Architecture-level; worth a look after design 1.
- Prefix caching on this hybrid hits only at 2304–3584-token page boundaries in align mode; `--prefix-match-unit` and the retention interval control it; `reasoning_effort` changes flush it (#111, fixed upstream in #63).
- NCCL effective ~10 GB/s without GPUDirect RDMA; allreduce is ~7% of the step.

## 7. Open questions the research could not settle

- The realistic `mma.sync` fp16 efficiency on GB10 with 99 KB smem (assumed 60–75% of 100 TFLOPS). Design 1's magnitude depends on it; the first kernel prototype answers this.
- Whether PR77's E2 kernels already stage A rows in shared memory (then design 1 is an extension) or still re-decode per 16 rows (then it is a replacement). Read `exl3_fat_gemm.cu` in the new image first.
- KDA backend actually used in our image on sm_121 (Triton FLA assumed; 6% of the step for 0.7% of FLOPs).
