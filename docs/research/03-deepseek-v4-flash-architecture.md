# DeepSeek-V4-Flash vs GLM-5.3-Flash: why V4-Flash prefills ~2× faster on 2× DGX Spark

Legend: **[F]** = fact with citation; **[I]** = my inference/arithmetic.

## 1. DeepSeek-V4-Flash architecture

**[F]** From `config.json` ([deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)) and the tech report §4.2.1 ([arXiv 2606.19348](https://arxiv.org/abs/2606.19348)):

| Field | Value | Meaning |
|---|---|---|
| params | 284B total / 13B active | paper §4.2.1 |
| `num_hidden_layers`, `hidden_size` | 43, 4096 | |
| `compress_ratios` | `[0,0,4,128,4,128,…,4,0]` | layers 0,1,42 = pure sliding-window; 20 CSA (m=4) and 20 HCA (m′=128) interleaved |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 64 / 1 / 512 | shared K=V MQA: one 512-d entry per (compressed) position, `qk_rope_head_dim`=64 of it RoPE'd (BF16 in cache), rest FP8 |
| `q_lora_rank` | 1024 | latent query d_c, shared by attention and indexer queries |
| `o_groups`, `o_lora_rank` | 8, 1024 | grouped low-rank output projection (paper: "c·n_h is quite large") |
| `sliding_window` | 128 | extra uncompressed local branch in every layer; learnable attention sink |
| `index_n_heads`, `index_head_dim`, `index_topk` | 64, 128, **512** | Lightning indexer over 4:1-compressed keys, ReLU-weighted dot products (eqs 13–17); top-k is 512 compressed entries (V3.2 was 2048 raw tokens, [V3.2 config](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json)) |
| `n_routed_experts`, `num_experts_per_tok`, `n_shared_experts`, `moe_intermediate_size` | 256, 6, 1, 2048 | |
| `scoring_func`, `topk_method`, `routed_scaling_factor` | `sqrtsoftplus`, `noaux_tc`, 1.5 | Sqrt(Softplus) replaces sigmoid; bias-based aux-loss-free balancing; no `n_group/topk_group` |
| `num_hash_layers` | 3 | first 3 MoE layers route by a frozen token-id→expert table (no dense FFN layers at all) — [HF transformers doc](https://huggingface.co/docs/transformers/model_doc/deepseek_v4) |
| `num_nextn_predict_layers` | 1 | MTP depth 1; `-0731` adds DSpark (`dspark_block_size` 5, target layers 40–42) — [0731 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json) |
| `expert_dtype` | `fp4` | routed experts are **MXFP4** (E2M1, e8m0 scale per 32); everything else FP8 e4m3 with `weight_block_size [128,128]`, `scale_fmt ue8m0`, dynamic per-token-group activations |
| `hc_mult`, `hc_sinkhorn_iters` | 4, 20 | mHC: 4 residual streams |
| `max_position_embeddings` | 1,048,576 | YaRN ×16 from 65,536 |

**[F]** Paper §2.3.4: KV stored as BF16 RoPE dims + FP8 rest; indexer QK path runs in **FP4**; index scores in BF16 (2× faster top-k, 99.7% recall, §5.2.1). At 1M context Flash needs "only 10% of the single-token FLOPs and 7% of the KV cache" of V3.2. FlashMLA's `fp8_ds_mla` record is 512 B FP8 + 16 B scales + 128 B BF16 RoPE = 656 B/token-entry ([FlashMLA](https://github.com/deepseek-ai/FlashMLA)).

**Correction to your framing [F]:** V4-Flash is not "FP8 weights" — 96% of parameters (routed experts) are native MXFP4; sparkrun's README itself notes the "NVFP4" alias "is actually native FP8" + mxfp4 MoE ([tonyd2wild](https://github.com/tonyd2wild/deepseek-v4-flash-dgx-spark)).

## 2. Why prefill is efficient — per-token cost model

**[I]** Per-token projection parameters from config fields (2 FLOPs/param):

| Component | V4-Flash | GLM-5.3-Flash |
|---|---|---|
| Attention projections / layer | 118 M (q_down 4M + q_up 33.5M + KV/Z 4.2M + indexer 9M + grouped O 67M) | KDA: 134 M qkvo (+~17 M gates); DSA: 124 M (q_lora 1536→64×256, kv_lora 512→64×512, O 67M, indexer 32×128) |
| MoE active / layer | 7 experts × 25.2 M = 176 M | 9 experts × 25.2 M = 227 M (42 layers) + 3 dense FFN × 151 M |
| Active params (incl. LM head) | **≈13.2 B → ≈26 GFLOPs/token** | **≈17.1 B → ≈34 GFLOPs/token** |
| Routed-expert weights on device | 277 B params × (0.5+1/32) B ≈ **147 GB** MXFP4 | 304 B params ≈ **152 GB** at 4 bpw EXL3 (306 GB FP8) |
| Attention core / token @32K ctx | CSA 84 MF/layer ×20 + HCA 50 MF ×20 ≈ 2.7 GF; FP4 indexer ≈ 2.7 GF | DSA 268 MF ×11 ≈ 3.0 GF; indexer ≈ 0.7 GF; KDA chunkwise ≈ 0.3 GF nominal |
| Attention core / token @100K | ≈ 4.1 GF + 8.4 GF FP4 indexer | ≈ 3.0 + 2.3 + 0.3 GF |

Key observations:

- **Weight streaming is not the limiter for either model at 8K chunks [I].** TP2 gives each GPU ~74–76 GB of experts; at 273 GB/s that is ~0.27 s per chunk, vs. a measured chunk time of 8192/1900 ≈ 4.3 s (V4) or ≈9 s (GLM). Prefill is compute/kernel-bound on both.
- **Sustained tensor-core utilisation is nearly identical once you normalise by datapath precision [I].** 1,900 tok/s × ~32 GFLOPs ≈ 61 TFLOP/s = ~30 TFLOP/s per GB10 ≈ 15% of the measured FP8 GEMM peak (mamf-finder: **207.7 TFLOPS FP8, 99.8 TFLOPS BF16** — [StorageReview](https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops)). GLM: 900 × ~40 GFLOPs ≈ 36 TFLOP/s = ~18 per GPU ≈ 18% of the BF16/FP16 peak. So the gap decomposes as ≈1.3× (active FLOPs) × ≈2× (FP8/FP4 vs FP16 GEMM ceiling) ÷ ~1.2 (GLM slightly better utilised) ≈ 2.1×.
- **Expert granularity is not a differentiator [I].** Both use 2048-wide experts; tokens/expert/chunk at 8192 tokens = 8192×6/256 = 192 (V4) vs 8192×8/288 = 228 (GLM). GEMM M-dimension per expert is the same order.
- **DeepGEMM/FlashMLA on SM12x [F]:** DeepGEMM PR #324 (merged 2026-06-24) adds `sm120_fp8_fp4_gemm_1d1d`, M-grouped contiguous/masked grouped GEMMs, `sm120_fp8/fp4_mqa_logits` (paged, L2-cached KV), and `sm120_tf32_hc_prenorm_gemm` for mHC; on RTX PRO 6000 it reaches 96% of roofline (778 TFLOPS FP8, 1561 FP4) and 1049 TFLOPS ragged-FP4 MQA logits ([PR #324](https://github.com/deepseek-ai/DeepGEMM/pull/324)). MegaMoE stays SM100-only ([vllm-gb10 #26](https://github.com/timothystewart6/vllm-gb10/issues/26)). Contiguous layout requires expert segments aligned to the M block; masked layout serves CUDA-graph decode ([DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM)). FlashMLA sparse prefill hits 640 TFLOPS on H800 / 1450 on B200 but is SM90/SM100 only; on GB10 the sparse-MLA path is FlashInfer's `sparse_mla_sm120` ([vLLM PR #41834](https://github.com/vllm-project/vllm/pull/41834)).
- **Weight format [F/I]:** EXL3's GEMM takes fp16 activations and, above a small M, reconstructs trellis weights before the matmul ("the reconstruct threshold keeps m <= 144" — [exl3_gemm.cu](https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/exllamav3_ext/quant/exl3_gemm.cu)); the fat-expert PR77 kernels gave only +20% ([MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)). Nothing in that path touches FP8/FP4 tensor cores. General finding: W4A16 "gets nothing" in compute-bound prefill and pays dequant; W4A8 is up to 58% faster prefill than W4A16 ([SqueezeBits](https://blog.squeezebits.com/vllm-vs-tensorrtllm-7-weightactivation-quantization-34461), [dev.to](https://dev.to/ji_ai/why-int4-weight-only-quantization-doesnt-speed-up-prefill-1b45)).
- **Linear-attention kernels [F]:** FlashKDA (CUTLASS) requires SM90+ and is 1.7–2.2× faster than the FLA Triton kernel on H20 ([FlashKDA](https://github.com/MoonshotAI/FlashKDA)); vLLM runs FlashKDA for prefill where available and FLA/Triton otherwise ([vLLM K3 blog](https://vllm.ai/blog/2026-07-27-k3)). **[I]** GB10 coverage is undocumented; assume Triton FLA on sm_121. Nominal KDA FLOPs are tiny (Kimi Linear: `6Td_h² + 3TCd_h + TC²` per head, [arXiv 2510.26692](https://arxiv.org/abs/2510.26692)), so the cost is utilisation, not FLOPs; Kimi Linear only pulls ahead of MLA at ≥128K context.

## 3. GB10 / DGX Spark DeepSeek-V4 recipe

**[F]** Stack: stock vLLM does not run V4 on SM12x; the community uses the `jasl/vllm` fork of **PR #41834** ("Add SM12x support for DeepSeek V4 Flash", open, 288 commits, tags `sm120-pr-41834-stable-preview-2026MMDD`), env `VLLM_DEEPSEEK_V4_FLASHINFER_SM120_{DECODE,PREFILL}=1` (default on; `=0` falls back to Triton), V2 model runner, `TORCH_CUDA_ARCH_LIST=12.1a` ([hazyumps](https://github.com/hazyumps/deepseek-v4-flash-gb10), [PR #41834](https://github.com/vllm-project/vllm/pull/41834)). Images: `hazyumps/deepseek-v4-flash-gb10:sm121-cu130-20260727d`, `aidendle94/sparkrun-vllm-ds4-gb10:production-v2` (`VLLM_USE_B12X_MOE=1` — FlashInfer b12x fused MoE from [vLLM PR #40082](https://github.com/vllm-project/vllm/pull/40082)). Kernel enablement gates: FlashInfer sparse-MLA SM120 (PR 3395, ≥0.6.13), DeepGEMM #324; residual gaps tracked in [vLLM #41063](https://github.com/vllm-project/vllm/issues/41063) (FP4 attention/einsum). Startup log "Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM" confirms the native path; TileLang and Triton caches are both present.

**[F]** Flags in use: `--tensor-parallel-size 2 --nnodes 2 --distributed-executor-backend mp`, `--kv-cache-dtype fp8`, `--block-size 256`, `--max-num-batched-tokens 8192`, `--max-num-seqs 2–8`, `--gpu-memory-utilization 0.8–0.835`, `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` (0731: `dspark`, n must equal `dspark_block_size`=5), `--enable-prefix-caching`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256`, NCCL 2.30.4 ([tonyd2wild 1M](https://github.com/tonyd2wild/deepseek-v4-flash-2x-spark-1m), [dual-spark](https://github.com/tonyd2wild/deepseek-v4-flash-dual-spark-recipe)).

**[F]** Reported numbers (2× GB10, TP2): prefill ~1.6–1.8k tok/s (hazyumps), ~1,785 tok/s and ~1,380 at 500K ([elsung](https://github.com/elsung/dgx-spark-deepseek-v4-flash)), 924 tok/s on a 400K prompt / 786 on 800K, 33K in 31.9 s; decode 40–47 tok/s (MTP n=2, ~78% acceptance on code), 45.5 tok/s unchanged 500K→1M; aggregate 92 tok/s @8, ~350 @32; syncing driver/firmware across nodes was "+140% prefill". Known issues: MTP + FULL CUDA graphs token corruption (fixed by vLLM #51318/#52492/#52836 — [NVIDIA forum](https://forums.developer.nvidia.com/t/deepseek-v4-flash-on-2x-dgx-spark-intermittent-token-corruption-with-mtp-cuda-graphs/380889)), prefix-cache leak (#44237), cold-start timeout, experimental 416 B NVFP4 KV (fall back to fp8 if gibberish — [forum](https://forums.developer.nvidia.com/t/native-416-byte-nvfp4-kv-cache-dspark-for-deepseek-v4-flash-0731-on-2x-dgx-spark/379788)). SGLang on SM12x: DSPARK topk-bucket crash ([#33134](https://github.com/sgl-project/sglang/issues/33134)); dsv4 prefill 2.8–5.5k tok/s vs vLLM ~12.5k on 4× RTX PRO 6000 ([#33422](https://github.com/sgl-project/sglang/issues/33422)).

## 4. FP8 activation quantisation: evidence

**[F]** DeepSeek-V3 report §3.3 ([arXiv 2412.19437](https://arxiv.org/html/2412.19437v2)): activations scaled per 1×128 tile, weights per 128×128 block, E4M3 everywhere, online (not delayed) scaling; FP32 promotion on CUDA cores every N_C=128 elements because H800 tensor cores keep only 14 mantissa bits; "relative loss error … consistently below 0.25%" vs BF16 on ~16B and ~230B models over ~1T tokens; embedding, LM head, MoE gating, norms and attention kept high precision; block-wise (128×128) *activation* scaling caused instability (App. B.2). **[F]** V4 §5.2.1: MXFP4 experts via QAT with FP32 master weights; FP4→FP8 dequant is lossless because 1×32 FP4 scales fit inside the 128×128 FP8 block's dynamic range; rollout uses native FP4. Unsloth verified bit-identical MXFP4 repacking (KL≈0) and measured post-hoc 4-bit of the *non-expert* tensors at KL 0.0102 / 96.3% top-token agreement ([unsloth](https://unsloth.ai/docs/models/deepseek-v4)).

**[I]** No engine implements "FP8 activations for prefill, weight-only for decode" as a switch; the practical equivalent is a W4A8 checkpoint (4-bit weights, FP8 activations) — decode is weight-bandwidth-bound so it costs nothing there, and prefill gets the FP8 tensor-core path. Community W4A8-FP8 V4-Flash checkpoints exist ([endnai](https://huggingface.co/endnai/DeepSeek-V4-Flash-W4A8-FP8)). No published KL/ppl for EXL3-4bpw GLM-5.3-Flash vs the FP8 native checkpoint was found.

## 5. Serving-side scheduling

**[F]** vLLM: chunked prefill (8192 on GB10), `--block-size 256` (paper: cache blocks span lcm(m, m′)=128 tokens), single-node DP4+EP ([vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)), fused compressor+RMSNorm+RoPE+cache-insert (1.4–3×), FP4 indexer cache ([vLLM blog](https://vllm-project.github.io/2026/04/24/deepseek-v4.html)). SGLang: DP/TP/CP attention, EP on DeepEP + MegaMoE (FP8×FP4 mega-kernel, SM100), FlashMLA fused SWA+compressed attention, radix-select top-k (~15 µs), TileLang mHC fusion, HiSparse CPU offload of C4 KV (up to 3× capacity), PD disaggregation via Mooncake/NIXL ([LMSYS](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/), [SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4)). **[I]** With two GPUs, DP-attention, CP and PD-disaggregation are inapplicable (each needs ≥2 model replicas or a full model per role). What transfers: chunk-size tuning, prefix caching, EP=2 (hazyumps runs TP2+EP; halves per-GPU expert weights but adds all-to-all over RoCE — a wash for prefill).

## Side-by-side

| | DeepSeek-V4-Flash | GLM-5.3-Flash ([config](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)) |
|---|---|---|
| Total / active | 284B / 13B | 320B / 18B |
| Layers | 43 (3 SWA, 20 CSA, 20 HCA) | 45 (34 KDA + 11 NoPE sparse-MLA), 3 dense FFN |
| Attention | MQA, 1 KV head × 512, 64 q-heads, RoPE 64 | KDA 64×128 (conv 4); MLA `kv_lora_rank` 512, `qk_nope` 256, `v` 256, `qk_rope` 0, 64 heads |
| Sparse selection | 64 idx heads ×128, top-512 over 4:1 compressed keys, FP4 | 32 idx heads ×128, top-2048, `index_kpool` 4 |
| MoE | 256 exp, top-6 +1 shared, 2048, sqrtsoftplus, scale 1.5 | 288 exp, top-8 +1 shared, 2048, sigmoid, scale 2.5, `n_group` 1 |
| Weights (as served) | FP8 128×128 + MXFP4 experts (native, QAT) | EXL3 4 bpw trellis (fp16 GEMM) / NVFP4 |
| Projection FLOPs/token | ≈26 GF | ≈34 GF |
| KV/token | ~656 B per CSA entry (÷4), ÷128 HCA, SWA 128 | KDA state (no KV) + 11 layers × 656 B `fp8_ds_mla` |
| GB10 attention kernels | FlashInfer sparse-MLA SM120 + DeepGEMM sm120 mqa logits | No upstream SM120 NoPE sparse-MLA ([vLLM #53963](https://github.com/vllm-project/vllm/issues/53963)); community kernels; KDA via Triton FLA |
| GB10 prefill reported | 1.6–1.8k tok/s | EXL3: 938–997 (pre-PR77) → 1,132–1,241 with fat-expert kernels; NVFP4: 428–1,283 ([Libertai](https://github.com/Libertai/glm53-flash-vllm-gb10)) |

## Implications for our setup (ranked)

1. **GEMM datapath precision — dominant, high confidence on direction, medium on magnitude (~1.5–2×).** V4 runs FP8 dense and FP8×MXFP4 expert GEMMs on paths with a 207 TFLOPS ceiling; EXL3 reconstructs to fp16 and runs against a ~100 TFLOPS ceiling. Both engines sit at ~15–18% of their respective ceilings. **Addressable:** move GLM to a W4A8/NVFP4 path that keeps activations 8-bit through prefill (`flashinfer_cutlass`/b12x NVFP4 MoE, fp8 activations). The native FP8 checkpoint (306 GB) does not fit 2×128 GB, so NVFP4 (181 GiB) or W4A8 is the realistic target; note the Libertai NVFP4 build needed `VLLM_GLM53_MOE_INPUT_SCALE=1.0` and a hand-written sparse-MLA kernel.
2. **Active FLOPs/token, ~1.3×, high confidence.** 18B vs 13B active (9 vs 7 experts/layer, dense FFN layers, larger vocab). Not addressable.
3. **Attention-kernel maturity on sm_121 (largest factor for 100K TTFT), medium confidence.** V4 gets FlashInfer's SM120 sparse-MLA plus DeepGEMM sm120 FP4 indexer kernels; GLM's rope-free sparse MLA has no upstream SM120 lane (three failure modes in #53963), and KDA prefill likely runs Triton FLA rather than FlashKDA. At 100K, V4 nominally does *more* attention FLOPs than GLM yet wins TTFT — kernels, not FLOPs. **Partially addressable:** port SM100's rope-free sparse-MLA lane to SM120 (the issue's proposed fix) or use SGLang's TileLang `dsa` backend; watch FlashKDA for SM12x.
4. **Engine-level, medium-high confidence, cheap.** Your 900 tok/s matches the pre-PR77 EXL3 baseline (938/997/941); the PR77 fat-expert kernels add +20%. The NVIDIA forum thread documents 2× prefill degradation from spec-decode state fragmenting the hybrid KDA+MLA cache (14.6 s → 27 s on 24K), fixed by disabling the drafter during long prefill; `--mixed-prefill-token-cap -1` and MNBT 4096 improved decode-during-prefill 5.6× and short-request TTFT 30% ([forum](https://forums.developer.nvidia.com/t/glm-5-3-flash-on-2x-gb10-speculative-decoding-makes-long-prefill-ttft-alternate-2x-after-a-mixed-workload-plus-3-knobs-that-measurably-helped/382099)). Also sync node firmware/driver (+140% for V4).
5. **Expert count/granularity — not a factor, high confidence.** Same expert width, near-identical tokens/expert/chunk.
6. **Weight-bandwidth — not a factor at 8K chunks (≤7% of chunk time).** Shrinking chunks below ~2K would make it one.

Related search sources also consulted: [DeepSeek V4 model-card PDF](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf), [DeepSeek-V3.2 report](https://arxiv.org/abs/2512.02556), [vLLM GLM-5.3-Flash recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash), [SGLang GLM-5.3-Flash](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash), [tonyd2wild GLM NVFP4](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark), [randomllama SGLang GLM DFlash2](https://huggingface.co/randomllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark), [sebastianraschka GLM notes](https://sebastianraschka.com/blog/2026/glm-5-3-flash-architecture-notes.html).
