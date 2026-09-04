#!/usr/bin/env python3
"""Generate the M-tiled EXL3 MoE patch set against exllamav3 @ c5d9c657.

Reads the pristine sources from exl3src/, applies exact-anchored edits, writes
the patched tree to exl3patch/<path-in-repo>. Every anchor is asserted, so a
silent no-op is impossible.

What it changes (see KERNEL_PLAN.md):
  exl3_gemm_inner.cuh   generalise TILESIZE_M (was hard-asserted 16): frag_a/frag_c
                        indexed by M-block, mma over m×n, reductions and C writes
                        loop over M-blocks, sh_c sized per M-block.
  exl3_moe_kernel.cuh   kernel templated on MOE_TM; GEMM loops step MOE_TM;
                        FRAG_STAGES 2 when MOE_TM > 16 (register budget).
  exl3_moe.cu           second instance table (M=64, N=128) selected when bsz > 16.
  exl3_moe_instances.cuh + two new instance .cu files (k0 and k4, n128, m64).
Decode (bsz <= 16) keeps the exact original kernel instances.
"""
import os, sys

SRC = "exl3src"
DST = "exl3patch/exllamav3/exllamav3_ext/quant"
os.makedirs(f"{DST}/comp_units", exist_ok=True)


def load(name):
    return open(f"{SRC}/{name}").read()


def rep(s, old, new, n=1, tag=""):
    c = s.count(old)
    assert c == n, f"[{tag}] anchor count {c} != {n}: {old[:80]!r}"
    return s.replace(old, new)


# ------------------------------------------------------------- exl3_gemm_inner.cuh
s = load("exl3_gemm_inner.cuh")
T = "gemm_inner"

s = rep(s,
        '    static_assert(TILESIZE_M == 16, "Invalid kernel params");                     // strictly assume size_m <= 16\n',
        '    static_assert(TILESIZE_M % 16 == 0 && TILESIZE_M <= 128, "Invalid kernel params");  // M-tiled: size_m <= TILESIZE_M\n',
        tag=T)

# sh_c must hold one reduction row-set per M-block
s = rep(s,
        '    const int sh_c_size = MAX  // in floats\n    (\n        4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP,\n',
        '    const int sh_c_size = MAX  // in floats\n    (\n        4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP * TILEBLOCKS_M,\n',
        tag=T)

# fragments: one A fragment and one C accumulator per M-block
s = rep(s,
        '    register FragA frag_a[FRAG_STAGES];\n    register FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];\n    register FragC frag_c[FRAGS_N_PER_WARP];\n',
        '    register FragA frag_a[FRAG_STAGES][TILEBLOCKS_M];\n    register FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];\n    register FragC frag_c[TILEBLOCKS_M][FRAGS_N_PER_WARP];\n',
        tag=T)

# load_frags: the m loop already exists; index the fragment instead of overwriting one
s = rep(s,
        '                ldsm4(frag_a[buf], (int4*) sh1_a_ptr + R * A_COLS + c_swizzled);\n',
        '                ldsm4(frag_a[buf][m], (int4*) sh1_a_ptr + R * A_COLS + c_swizzled);\n',
        tag=T)

# clear
s = rep(s,
        '        #pragma unroll\n        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)\n            frag_c[n] = {};\n',
        '        #pragma unroll\n        for (int m = 0; m < TILEBLOCKS_M; ++m)\n            #pragma unroll\n            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)\n                frag_c[m][n] = {};\n',
        tag=T)

# reduction store/add: FRAGS_N*4 floats per thread per M-block
s = rep(s,
        '''        auto store = [&] (int i)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[n][j];
                }
            }
            __syncthreads();
        };

        auto add = [&] (int i)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) frag_c[n][j] += *sh_red++;
                }
            }
        };
''',
        '''        auto store = [&] (int i)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4 * TILEBLOCKS_M) * t;
                #pragma unroll
                for (int m = 0; m < TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[m][n][j];
                }
            }
            __syncthreads();
        };

        auto add = [&] (int i)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4 * TILEBLOCKS_M) * t;
                #pragma unroll
                for (int m = 0; m < TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) frag_c[m][n][j] += *sh_red++;
                }
            }
        };
''',
        tag=T)

# small (<=8 rows) fast path only touches M-block 0 — keep it, but only when the tile IS one block
s = rep(s,
        '''                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    *sh_red++ = frag_c[n][0];
                    *sh_red++ = frag_c[n][1];
                }
            }
            __syncthreads();
        };
''',
        '''                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    *sh_red++ = frag_c[0][n][0];
                    *sh_red++ = frag_c[0][n][1];
                }
            }
            __syncthreads();
        };
''',
        tag=T)
s = rep(s,
        '''                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    frag_c[n][0] += *sh_red++;
                    frag_c[n][1] += *sh_red++;
                }
            }
        };
''',
        '''                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    frag_c[0][n][0] += *sh_red++;
                    frag_c[0][n][1] += *sh_red++;
                }
            }
        };
''',
        tag=T)
s = rep(s, '        if (size_m <= 8)\n        {\n            if constexpr (TILEBLOCKS_K == 2)\n',
        '        if (TILEBLOCKS_M == 1 && size_m <= 8)\n        {\n            if constexpr (TILEBLOCKS_K == 2)\n', tag=T)

# write_sum_tile_sh (shmem hadamard output; standalone path) — loop M-blocks
s = rep(s,
        '''    auto write_sum_tile_sh = [&]()
    {
        const int n0 = warp_id * FRAGS_N_PER_WARP;
        const int r0 = lane_id / 4;
        const int r1 = r0 + 8;
        if (r0 < size_m)
        {
            const int c = (lane_id % 4) * 2;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                float* c_ptr = ((float*) sh_c) + r0 * TILESIZE_N + (n0 + n) * 8 + c;
                *c_ptr++ = frag_c[n][0];
                *c_ptr++ = frag_c[n][1];
            }
        }
        if (r1 < size_m)
        {
            const int c = (lane_id % 4) * 2;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                float* c_ptr = ((float*) sh_c) + r1 * TILESIZE_N + (n0 + n) * 8 + c;
                *c_ptr++ = frag_c[n][2];
                *c_ptr++ = frag_c[n][3];
            }
        }
    };
''',
        '''    auto write_sum_tile_sh = [&]()
    {
        const int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        {
            const int r0 = lane_id / 4 + 16 * m;
            const int r1 = r0 + 8;
            if (r0 < size_m)
            {
                const int c = (lane_id % 4) * 2;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    float* c_ptr = ((float*) sh_c) + r0 * TILESIZE_N + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[m][n][0];
                    *c_ptr++ = frag_c[m][n][1];
                }
            }
            if (r1 < size_m)
            {
                const int c = (lane_id % 4) * 2;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    float* c_ptr = ((float*) sh_c) + r1 * TILESIZE_N + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[m][n][2];
                    *c_ptr++ = frag_c[m][n][3];
                }
            }
        }
    };
''',
        tag=T)

# read_sum_gl — loop M-blocks
s = rep(s,
        '''    auto read_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            int r0 = lane_id / 4;
            int r1 = r0 + 8;
            int c = (lane_id % 4) * 2;
            if (r0 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                    frag_c[n][0] += *c_ptr++;
                    frag_c[n][1] += *c_ptr++;
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                    float2 interm = __half22float2(*c_ptr);
                    frag_c[n][0] += interm.x;
                    frag_c[n][1] += interm.y;
                }
            }
            if (r1 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                    frag_c[n][2] += *c_ptr++;
                    frag_c[n][3] += *c_ptr++;
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                    float2 interm = __half22float2(*c_ptr);
                    frag_c[n][2] += interm.x;
                    frag_c[n][3] += interm.y;
                }
            }
        }
    };
''',
        '''    auto read_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            int r0 = lane_id / 4 + 16 * m;
            int r1 = r0 + 8;
            int c = (lane_id % 4) * 2;
            if (r0 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                    frag_c[m][n][0] += *c_ptr++;
                    frag_c[m][n][1] += *c_ptr++;
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                    float2 interm = __half22float2(*c_ptr);
                    frag_c[m][n][0] += interm.x;
                    frag_c[m][n][1] += interm.y;
                }
            }
            if (r1 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                    frag_c[m][n][2] += *c_ptr++;
                    frag_c[m][n][3] += *c_ptr++;
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                    float2 interm = __half22float2(*c_ptr);
                    frag_c[m][n][2] += interm.x;
                    frag_c[m][n][3] += interm.y;
                }
            }
        }
    };
''',
        tag=T)

# write_sum_gl — loop M-blocks
s = rep(s,
        '''    auto write_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            int r0 = lane_id / 4;
            int r1 = r0 + 8;
            int c = (lane_id % 4) * 2;
            if (r0 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[n][0];
                    *c_ptr++ = frag_c[n][1];
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                    half2 sum = __floats2half2_rn(frag_c[n][0], frag_c[n][1]);
                    *c_ptr = sum;
                }
            }
            if (r1 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[n][2];
                    *c_ptr++ = frag_c[n][3];
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                    half2 sum = __floats2half2_rn(frag_c[n][2], frag_c[n][3]);
                    *c_ptr = sum;
                }
            }
        }
    };
''',
        '''    auto write_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            int r0 = lane_id / 4 + 16 * m;
            int r1 = r0 + 8;
            int c = (lane_id % 4) * 2;
            if (r0 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[m][n][0];
                    *c_ptr++ = frag_c[m][n][1];
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                    half2 sum = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                    *c_ptr = sum;
                }
            }
            if (r1 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[m][n][2];
                    *c_ptr++ = frag_c[m][n][3];
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                    half2 sum = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
                    *c_ptr = sum;
                }
            }
        }
    };
''',
        tag=T)

# matmul: B fragment reused across all M-blocks — the entire point of the patch
s = rep(s,
        '''    auto matmul = [&] (int buf)
    {
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            ptx_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c[n]);
    };
''',
        '''    auto matmul = [&] (int buf)
    {
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                ptx_mma_m16n8k16(frag_a[buf][m], frag_b[buf][n], frag_c[m][n]);
    };
''',
        tag=T)

open(f"{DST}/exl3_gemm_inner.cuh", "w").write(s)
print("wrote exl3_gemm_inner.cuh")

# ------------------------------------------------------------- exl3_moe_kernel.cuh
s = load("exl3_moe_kernel.cuh")
T = "moe_kernel"
s = rep(s,
        'template<int t_bits, int MOE_TILESIZE_N>\n__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)\nvoid exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)\n{\n',
        '// MOE_TM = rows per GEMM pass. 16 = original decode kernel (B fragments used once).\n'
        '// 64 = prefill kernel: each dequantised B fragment serves four 16-row A blocks.\n'
        '// MIN_BLOCKS = 2 asks the compiler to fit two blocks per SM (<=64 regs/thread): the\n'
        '// baseline profiles at 33%% occupancy with schedulers idle 55%% of cycles (stall-bound).\n'
        '// MOE_SH / MOE_FSX: cp.async pipeline depth and register fragment stages. Baseline is 3/3\n'
        '// and profiles with 45M local-memory spill requests per launch and 31%% of stall time at\n'
        '// the CTA barrier: fewer register stages remove spills, deeper smem stages cut waits.\n'
        'template<int t_bits, int MOE_TILESIZE_N, int MOE_TM = MOE_TILESIZE_M, int MIN_BLOCKS = 1, int MOE_SH = MOE_SH_STAGES, int MOE_FSX = 0>\n'
        '__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16, MIN_BLOCKS)\n'
        'void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)\n{\n'
        '    constexpr int MOE_FS = MOE_FSX ? MOE_FSX : ((MOE_TM > 16) ? 2 : MOE_FRAG_STAGES);\n',
        tag=T)
# both GEMM lambdas: shape args + loop step
s = rep(s,
        '                #define SHAPE_ARGS      \\\n                    MOE_TILESIZE_M,     \\\n                    MOE_TILESIZE_K,     \\\n                    MOE_TILESIZE_N,     \\\n                    MOE_SH_STAGES,      \\\n                    MOE_FRAG_STAGES\n',
        '                #define SHAPE_ARGS      \\\n                    MOE_TM,             \\\n                    MOE_TILESIZE_K,     \\\n                    MOE_TILESIZE_N,     \\\n                    MOE_SH,             \\\n                    MOE_FS\n',
        n=2, tag=T)
s = rep(s, '                    MIN(size_m, 16),    \\\n', '                    MIN(size_m, MOE_TM),\\\n', n=2, tag=T)
s = rep(s,
        '                in_addr += 16 * hidden_dim;\n                out_addr += 16 * intermediate_dim;\n                size_m -= 16;\n',
        '                in_addr += MOE_TM * hidden_dim;\n                out_addr += MOE_TM * intermediate_dim;\n                size_m -= MOE_TM;\n',
        tag=T)
s = rep(s,
        '                in_addr += 16 * intermediate_dim;\n                out_addr += 16 * hidden_dim;\n                size_m -= 16;\n',
        '                in_addr += MOE_TM * intermediate_dim;\n                out_addr += MOE_TM * hidden_dim;\n                size_m -= MOE_TM;\n',
        tag=T)
open(f"{DST}/exl3_moe_kernel.cuh", "w").write(s)
print("wrote exl3_moe_kernel.cuh")

# ------------------------------------------------------------- instances
s = load("exl3_moe_instances.cuh")
s = rep(s,
        'EXL3_MOE_DECLARE_GETTERS(8);\n',
        'EXL3_MOE_DECLARE_GETTERS(8);\n\n// Prefill (M-tiled) instances: M in {32,64} x N in {128,256}, runtime-K and 4-bit.\n'
        + "".join(f'fp_exl3_moe_kernel exl3_moe_kernel_k{k}_n{n}_m{m}();\n' for k, n, m in [(4,128,32),(4,128,64),(4,256,32),(4,256,64),(0,128,32),(0,128,64)])
        + 'fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m16_occ2();\nfp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m32_occ2();\n'
        + "".join(f'fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m16_s{sh}f{fs}();\n' for sh, fs in ((3,2),(4,3),(6,3),(6,2),(8,2))),
        tag="instances")
open(f"{DST}/comp_units/exl3_moe_instances.cuh", "w").write(s)
# k0 = runtime bit width: instantiates bits 1..8 and the 8-bit B stages + M-tiled reduction
# scratch exceed SMEM_MAX at N=256 — GLM is uniform 4-bit and takes the k4 instance, so k0
# exists only for the shapes that fit.
SHAPES = [(4, 128, 32), (4, 128, 64), (4, 256, 32), (4, 256, 64), (0, 128, 32), (0, 128, 64)]
for k, n, m in SHAPES:
    open(f"{DST}/comp_units/exl3_moe_inst_{k}_{n}_m{m}.cu", "w").write(
        '#include "exl3_moe_instances.cuh"\n#include "../exl3_moe_kernel.cuh"\n\n'
        f'fp_exl3_moe_kernel exl3_moe_kernel_k{k}_n{n}_m{m}() {{ return exl3_moe_kernel<{k}, {n}, {m}>; }}\n')
# pipeline-depth variants of the proven shape (k4, N=256, M=16, 1 block/SM): (SH_STAGES, FRAG_STAGES)
for sh, fs in ((3, 2), (4, 3), (6, 3), (6, 2), (8, 2)):
    open(f"{DST}/comp_units/exl3_moe_inst_4_256_m16_s{sh}f{fs}.cu", "w").write(
        '#include "exl3_moe_instances.cuh"\n#include "../exl3_moe_kernel.cuh"\n\n'
        f'fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m16_s{sh}f{fs}() {{ return exl3_moe_kernel<4, 256, 16, 1, {sh}, {fs}>; }}\n')
# occupancy variants of the ORIGINAL geometry (M=16) and M=32, N=256, 4-bit: 2 blocks/SM
for m in (16, 32):
    open(f"{DST}/comp_units/exl3_moe_inst_4_256_m{m}_occ2.cu", "w").write(
        '#include "exl3_moe_instances.cuh"\n#include "../exl3_moe_kernel.cuh"\n\n'
        f'fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m{m}_occ2() {{ return exl3_moe_kernel<4, 256, {m}, 2>; }}\n')
print("wrote instances")

# ------------------------------------------------------------- exl3_moe.cu (host dispatch)
s = load("exl3_moe.cu")
T = "moe_host"
s = rep(s,
        '    exl3_moe_kernel_k8_n128(), exl3_moe_kernel_k8_n256()\n};\n',
        '    exl3_moe_kernel_k8_n128(), exl3_moe_kernel_k8_n256()\n};\n\n'
        '// Prefill instances: 64 rows per GEMM pass so each dequantised weight fragment is\n'
        '// reused across four M blocks instead of being re-read and re-decoded per 16 rows.\n'
        '// Selected when the batch is larger than one decode tile. Env EXL3_MOE_PREFILL_M=16\n'
        '// forces the original path (A/B switch for validation).\n'
        'static fp_exl3_moe_kernel exl3_moe_prefill_kernel(int K, int N_off)\n{\n'
        '    // EXL3_MOE_PREFILL_M: 16 (original path) | 32 | 64 (default 64). EXL3_MOE_PREFILL_N: 128 | 256\n'
        '    // (default: same tile N the decode kernel would pick for these dims).\n'
        '    static int M = -1, N = -1;\n'
        '    if (M < 0) { const char* e = getenv("EXL3_MOE_PREFILL_M"); M = e ? atoi(e) : 64; }\n'
        '    if (N < 0) { const char* e = getenv("EXL3_MOE_PREFILL_N"); N = e ? atoi(e) : 0; }\n'
        '    int n = N ? N : (N_off ? 256 : 128);\n'
        '    static int occ2 = -1;\n'
        '    if (occ2 < 0) { const char* e = getenv("EXL3_MOE_OCC2"); occ2 = (e && atoi(e)) ? 1 : 0; }\n'
        '    static const char* st = getenv("EXL3_MOE_STAGES");\n'
        '    if (st && K == 4 && n == 256 && M == 16)\n    {\n'
        '        if (!strcmp(st, "s3f2")) return exl3_moe_kernel_k4_n256_m16_s3f2();\n'
        '        if (!strcmp(st, "s4f3")) return exl3_moe_kernel_k4_n256_m16_s4f3();\n'
        '        if (!strcmp(st, "s6f3")) return exl3_moe_kernel_k4_n256_m16_s6f3();\n'
        '        if (!strcmp(st, "s6f2")) return exl3_moe_kernel_k4_n256_m16_s6f2();\n'
        '        if (!strcmp(st, "s8f2")) return exl3_moe_kernel_k4_n256_m16_s8f2();\n'
        '    }\n'
        '    if (occ2 && K == 4 && n == 256 && M == 16) return exl3_moe_kernel_k4_n256_m16_occ2();\n'
        '    if (occ2 && K == 4 && n == 256 && M == 32) return exl3_moe_kernel_k4_n256_m32_occ2();\n'
        '    if (M == 16) return nullptr;\n'
        '    if (K == 4 && n == 128 && M == 32) return exl3_moe_kernel_k4_n128_m32();\n'
        '    if (K == 4 && n == 128 && M == 64) return exl3_moe_kernel_k4_n128_m64();\n'
        '    if (K == 4 && n == 256 && M == 32) return exl3_moe_kernel_k4_n256_m32();\n'
        '    if (K == 4 && n == 256 && M == 64) return exl3_moe_kernel_k4_n256_m64();\n'
        '    if (K == 0 && n == 128 && M == 32) return exl3_moe_kernel_k0_n128_m32();\n'
        '    if (K == 0 && n == 128 && M == 64) return exl3_moe_kernel_k0_n128_m64();\n'
        '    // (runtime-K at N=256 does not fit shared memory when M-tiled; falls back to the original path)\n'
        '    return nullptr;\n}\n',
        tag=T)
s = rep(s,
        '    int N_off = 0;\n    if (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0) N_off = 1;\n    fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[2 * K + N_off];\n',
        '    int N_off = 0;\n    if (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0) N_off = 1;\n    fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[2 * K + N_off];\n'
        '    if (bsz > 16)\n    {\n        fp_exl3_moe_kernel pk = exl3_moe_prefill_kernel(K, N_off);\n        if (pk) kernel = pk;\n    }\n'
        '    // Dynamic smem: the launcher used to request SMEM_MAX for every block, which alone caps\n'
        '    // occupancy at one block per SM. EXL3_MOE_SMEM_FIT=1 requests the shape\x27s real need.\n'
        '    static int smem_fit = -1;\n'
        '    if (smem_fit < 0) { const char* e = getenv("EXL3_MOE_SMEM_FIT"); smem_fit = (e && atoi(e)) ? 1 : 0; }\n'
        '    int smem = SMEM_MAX;\n'
        '    if (smem_fit)\n    {\n'
        '        static int pm = -1; if (pm < 0) { const char* e = getenv("EXL3_MOE_PREFILL_M"); pm = e ? atoi(e) : 64; }\n'
        '        int tm = (bsz > 16 && kernel != exl3_moe_kernel_instances[2 * K + N_off]) ? pm : 16;\n'
        '        int tn = N_off ? 256 : 128;\n'
        '        int kb = K ? K : 8;\n'
        '        int sh_a = tm * MOE_TILESIZE_K * 2;                                   // halfs -> bytes\n'
        '        int sh_b = (MOE_TILESIZE_K / 16) * (tn / 16) * 256 / 16 * kb * 2;      // uint16 -> bytes\n'
        '        int frags_n = 2 * (tn / 16) / (EXL3_GEMM_BASE_THREADS / 32);\n'
        '        int sh_c = 4 * EXL3_GEMM_BASE_THREADS * frags_n * (tm / 16) * 4;       // floats -> bytes\n'
        '        int sh_stages = MOE_SH_STAGES;\n'
        '        { const char* st = getenv("EXL3_MOE_STAGES"); if (st && st[0] == \x27s\x27) sh_stages = atoi(st + 1); }\n'
        '        smem = sh_stages * (sh_a + sh_b) + sh_c;\n'
        '    }\n',
        tag=T)
s = rep(s, '#include <set>\n', '#include <set>\n#include <cstdlib>\n#include <cstring>\n', tag=T)
s = rep(s, '        kernelArgs,\n        SMEM_MAX,\n        stream\n', '        kernelArgs,\n        smem,\n        stream\n', tag=T)
s = rep(s,
        'int exl3_moe_max_concurrency(int device)\n{\n    int num_sms = DevCtx::instance().get_num_sms(device);\n    return num_sms / MOE_SMS_PER_EXPERT;\n}\n',
        'int exl3_moe_max_concurrency(int device)\n{\n    int num_sms = DevCtx::instance().get_num_sms(device);\n'
        '    // EXL3_MOE_OCC2=1: two blocks per SM -> two expert groups per 8 SMs\n'
        '    const char* e = getenv("EXL3_MOE_OCC2"); int mult = (e && atoi(e)) ? 2 : 1;\n'
        '    return mult * num_sms / MOE_SMS_PER_EXPERT;\n}\n', tag=T)
s = rep(s, '    TORCH_CHECK(concurrency * MOE_SMS_PER_EXPERT <= num_sms, "Concurrency too high for device num_sms");\n',
        '    TORCH_CHECK(concurrency * MOE_SMS_PER_EXPERT <= 2 * num_sms, "Concurrency too high for device num_sms");\n', tag=T)
open(f"{DST}/exl3_moe.cu", "w").write(s)
print("wrote exl3_moe.cu")
print("patch tree ready under exl3patch/")
