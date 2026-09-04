# EXL3 prefill kernel plan — GLM-5.3-Flash on 2× GB10

Date: 2026-09-01. Source of truth: exllamav3 @ `c5d9c657966ffeeaa9353f0cc899f18629da4a13`
(the exact commit the MiaAI image builds; only `patch_exl3_ext_aarch64.py` — a build stub — is
applied on top). Sources cached in `exl3src/`.

## The finding

The trellis GEMM that every routed expert runs through is hard-wired to **16 rows**:

```
exl3_gemm_inner.cuh:47   static_assert(TILESIZE_M == 16, "Invalid kernel params");  // strictly assume size_m <= 16
exl3_moe_kernel.cuh:107  while (size_m > 0) { exl3_gemm_kernel_inner<...>(..., MIN(size_m, 16), ...); size_m -= 16; }
```

Per 16-row pass the kernel `cp_async`-loads the expert's packed B tile from global memory and
dequantises it (`dq_dispatch`) into register fragments — then uses those fragments for exactly one
`mma_m16n8k16` per N-fragment and discards them. An expert holding `c` tokens therefore reads and
decodes its weights `ceil(c/16)` times. **Cost is linear in tokens: this is a batched GEMV, not a
GEMM.** That is why chunk size never amortised in MiaAI's ladder, why letting 1024-row experts into
the kernel (P2b, 64 re-decodes) lost, and why 128-row tiling (P2a, 8 full walks) lost.

Roofline sanity check at MNBT=1024 (their profile: ~15 ms/layer fused): packed-weight read floor is
1.8 GB/node/layer ÷ 273 GB/s ≈ 6.6 ms; average ~28 rows/expert ⇒ ~2 passes ⇒ ~13 ms. Matches.
At MNBT=2048 (~57 rows ⇒ ~4 passes) the same kernel costs ~26 ms/layer — the ladder's flat line.

## Design decision: M-tiled trellis GEMM, not dequant-once

The obvious alternative (dequantise each expert once to fp16, run cuBLAS — what
`LinearEXL3.reconstruct_hgemm` does above `AUTO_RECONSTRUCT_THRESHOLD`, and MiaAI's "option 3")
is **wrong for this chip**: writing then re-reading fp16 is 7.2 GB/node/layer, 8× the packed read,
on a 273 GB/s part. The dequant itself is not the wall; the *repeated* read+dequant is.

The right fix keeps the trellis path and amortises it: **TILESIZE_M = 64** (TILEBLOCKS_M = 4).

Changes in `exl3_gemm_inner.cuh` (templated on TILESIZE_M so the decode path stays at 16):
1. Drop the `TILESIZE_M == 16` assert; `frag_a[FRAG_STAGES][TILEBLOCKS_M]`,
   `frag_c[TILEBLOCKS_M][FRAGS_N_PER_WARP]`.
2. `load_frags`: the existing `for m < TILEBLOCKS_M` loop already computes the swizzled row;
   index `frag_a[buf][m]` instead of overwriting one fragment.
3. `matmul`: `for m: for n: ptx_mma_m16n8k16(frag_a[buf][m], frag_b[buf][n], frag_c[m][n])`.
   B fragments are dequantised once per K-step and reused across all M blocks — the whole point.
4. Reduction/output (`threadblock_reduce`, `write_sum_gl`, `read_sum_gl`, `write_sum_tile_sh`):
   rows are `r0 = lane/4`, `r1 = r0+8`; generalise to `+ 16*m` with `row < size_m` predicates.
   The `size_m <= 8` fast path stays valid only for m == 0.
5. Shared memory: `sh_a` ×4 (3 stages × 64×32 halves = 12 KB); `sh_c` for the sub_k reduction
   scales with TILEBLOCKS_M: N=128 → 32 KB, N=256 → 64 KB. With `sh_b` 12 KB that is ~56 KB (N=128)
   or ~88 KB (N=256) against `SMEM_MAX` 90 KB. **Start with the N=128 instance.**
6. Registers: 512-thread blocks ⇒ ≤128 regs/thread. frag_c 4×2×4 = 32 (N=128), frag_a
   4×4×FRAG_STAGES. Use FRAG_STAGES = 2 for the M-tiled instance.
7. `exl3_moe_kernel.cuh`: loop step `TILESIZE_M` instead of 16; new instances
   `exl3_moe_kernel<bits, N, M=64>` alongside the existing M=16 ones; host dispatch picks M=64 when
   the batch is a prefill (tokens > 16).

Second, smaller inefficiency: expert→group assignment is static round-robin
(`expert_idx_assign++ % concurrency == group_idx`) regardless of token count, with concurrency = 6
on 48 SMs. Groups holding heavy experts finish last. Fix later with an atomic work counter.

Confirmed from source (line 56): the kernel skips experts with `count > max_tokens_per_expert`, so
the overlay's Python fat-expert pass does **not** double count. With M-tiling the fat cutoff can rise
(temps at 1024 rows × 6 groups × (4096+1024)×2 × 2 B ≈ 120 MB) and the Python fallback retires.

## Expected effect

MoE share of a prefill step is 63%. Passes per expert drop from ~4 (MNBT 2048) to ~1 ⇒ MoE ~3×
faster, and — unlike today — improving with larger chunks. Overall prefill 1.5–2× is the honest
target; the NVFP4/SGLang lane's 1,795 tok/s at 8K is the existence proof of what this model can do
on this hardware.

## Validation (rule 1, in order)

1. Op-level: one real MoE layer's expert tensors loaded from the checkpoint shards, real routing
   shapes (hidden 4096, intermediate_local 1024 at TP2, 288 experts, top-8, MNBT 1024/2048/4096,
   including 1024-row fat experts). Reference = `apply_exl3_python_loop` — an independent numeric
   path (LinearEXL3 → reconstruct + cuBLAS above the threshold). Assert max-abs / rel error at fp16.
2. Temp-0 output diff on a fixed prompt set vs the current path (maintenance window).
3. Needle ladder at 30K/100K/200K.
4. Only then: `bench_matrix.py` cold 32K/8K prefill for the number; then re-run the MNBT ladder.

## Build

Rebuild `exllamav3_ext` inside the image the way the Dockerfile does (L416–427): tarball of the
commit → `patch_exl3_ext_aarch64.py` → `TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=8 pip install`.
Iterate in a dev container on spark-head; ship as an overlay (patched source tree + rebuilt .so
bind-mounted), no image rebuild for serving.

## Blocked on

SSH to spark-head — Tailscale on the Mac is stopped.
