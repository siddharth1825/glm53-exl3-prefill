// exl3_grouped_prefill — design 1, grouped form: ONE launch per GEMM stage over ALL routed experts.
//
// Prefill MoE for EXL3 trellis experts without the per-expert Python loop:
//   1. rows sorted by expert (token_sorted, counts) — done by the overlay on the GPU
//   2. build_blocks:      per-expert M-block table from counts (single small kernel, no host sync)
//   3. gather_had13:      h13[r] = H128( x[token_sorted[r]] ⊙ suh_gate[e(r)] )        (fp16)
//   4. grouped_gemm13:    g|u[r] = ( h13[r] · W'_{e}[gate|up] ) H128 ⊙ svh              (fp16 out)
//   5. act_had2:          a[r]  = H128( (silu(min(g,lim)) * clamp(u,±lim)) ⊙ suh_down[e(r)] )
//   6. grouped_gemm2:     out[token[r]] += w[r] · ( a[r] · W'_{e}[down] ) H128 ⊙ svh     (fp32 atomicAdd)
// The GEMM mainloop is exl3_fat_gemm_v2's weight-stationary, cp.async-pipelined loop (bit-identical to
// PR77's exl3_fat_gemm), applied to every expert regardless of its row count. Blocks are (n-tile, m-block)
// pairs; a block whose table entry is past the end exits immediately, so the grid is an upper bound
// (rows/128 + E) known on the host without a sync.
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "quant/exl3_dq.cuh"
#include "quant/hadamard_inner.cuh"

namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int TILE_K = 64;
constexpr int STAGES = 2;
constexpr int KT = TILE_K / 16;
constexpr int M_BLOCKS = TILE_M / 16;
constexpr int N_TILES = TILE_N / 16;
constexpr int PACKED_WORDS = 64;
constexpr int A_STAGE_HALFS = TILE_M * TILE_K;
constexpr int A_ROW_INT4 = TILE_K / 8;
constexpr int B_STAGE_WORDS = KT * N_TILES * PACKED_WORDS;
constexpr int A_BYTES = STAGES * A_STAGE_HALFS * (int) sizeof(half);
constexpr int B_BYTES = STAGES * B_STAGE_WORDS * (int) sizeof(uint16_t);
constexpr int SMEM_TOTAL = A_BYTES + B_BYTES;        // 40 KB -> 2 CTAs per SM
constexpr float HAD_SCALE = 0.088388347648f;
constexpr int MAX_EXPERTS = 1024;

__device__ inline void had_ff_128_scale(const float* input_ptr, float* output_ptr, const half* scale)
{
    int lane = threadIdx.x & 31;
    float4 v = reinterpret_cast<const float4*>(input_ptr)[lane];
    float s0 = v.x + v.y, d0 = v.x - v.y, s1 = v.z + v.w, d1 = v.z - v.w;
    v.x = s0 + s1; v.y = d0 + d1; v.z = s0 - s1; v.w = d0 - d1;
    shuffle_had_f2x32(v.x, v.y, lane);
    shuffle_had_f2x32(v.z, v.w, lane);
    v.x *= HAD_SCALE; v.y *= HAD_SCALE; v.z *= HAD_SCALE; v.w *= HAD_SCALE;
    half4 sc = reinterpret_cast<const half4*>(scale)[lane];
    v.x *= __low2float(sc.x); v.y *= __high2float(sc.x); v.z *= __low2float(sc.y); v.w *= __high2float(sc.y);
    reinterpret_cast<float4*>(output_ptr)[lane] = v;
}

// ---------------------------------------------------------------------------------------------- 2. block table
// counts[E] -> for each block b: block_expert[b], block_row0[b] (row offset within the sorted rows);
// n_blocks written to meta[0]. One block of 1024 threads, E <= MAX_EXPERTS.
__global__ void build_blocks_kernel(const int32_t* __restrict__ counts, int n_experts, int max_blocks,
                                    int32_t* __restrict__ block_expert, int32_t* __restrict__ block_row0,
                                    int32_t* __restrict__ meta)
{
    __shared__ int32_t s_nblk[MAX_EXPERTS];
    __shared__ int32_t s_row0[MAX_EXPERTS];
    __shared__ int32_t s_blk0[MAX_EXPERTS];
    const int t = threadIdx.x;
    for (int e = t; e < n_experts; e += blockDim.x) s_nblk[e] = (counts[e] + TILE_M - 1) / TILE_M;
    __syncthreads();
    if (t == 0)    // serial scans over <= 1024 entries; trivial next to the GEMM
    {
        int row = 0, blk = 0;
        for (int e = 0; e < n_experts; ++e)
        {
            s_row0[e] = row; s_blk0[e] = blk;
            row += counts[e]; blk += s_nblk[e];
        }
        meta[0] = blk;
        meta[1] = row;
    }
    __syncthreads();
    for (int e = t; e < n_experts; e += blockDim.x)
    {
        for (int i = 0; i < s_nblk[e]; ++i)
        {
            const int b = s_blk0[e] + i;
            if (b < max_blocks) { block_expert[b] = e; block_row0[b] = s_row0[e] + i * TILE_M; }
        }
    }
    // mark the tail so GEMM blocks past the end exit
    const int nb = meta[0];
    for (int b = nb + t; b < max_blocks; b += blockDim.x) block_expert[b] = -1;
}

// ---------------------------------------------------------------------------------------------- 3./5. Hadamard passes
// One warp per (row, 128-block). expert_of_row[r] selects the per-expert sign vector.
// gather: src row = token_sorted[r] (fp16 [T, K]); otherwise src row = r.
template <bool gather>
__global__ __launch_bounds__(32)
void row_had_kernel(const half* __restrict__ src, half* __restrict__ dst, const int64_t* __restrict__ token_sorted,
                    const int32_t* __restrict__ expert_of_row, const half* __restrict__ suh, int64_t suh_stride,
                    int rows, int K)
{
    const int r = blockIdx.x;
    const int blk = blockIdx.y;
    if (r >= rows) return;
    const int64_t srow = gather ? token_sorted[r] : (int64_t) r;
    const half* in = src + srow * K + blk * 128;
    half* out = dst + (int64_t) r * K + blk * 128;
    const half* scale = suh + (int64_t) expert_of_row[r] * suh_stride;   // inner routine adds blockIdx.y*128 itself
    had_hf_r_128_inner<true, false>(in, out, scale, HAD_SCALE);
}

// gate|up (fp16 [rows, 2I]) -> act (fp16 [rows, I]) with the GLM swiglu clamp, then H128 ⊙ suh_down per row.
__global__ __launch_bounds__(32)
void act_had_kernel(const half* __restrict__ gu, half* __restrict__ act, const int32_t* __restrict__ expert_of_row,
                    const half* __restrict__ suh, int64_t suh_stride, int rows, int I, float limit)
{
    const int r = blockIdx.x;
    const int blk = blockIdx.y;
    if (r >= rows) return;
    const int lane = threadIdx.x;
    const half* g = gu + (int64_t) r * (2 * I) + blk * 128;
    const half* u = g + I;
    __shared__ __align__(16) half s_act[128];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const int c = lane * 4 + i;
        float gv = fminf(__half2float(g[c]), limit);
        float uv = fminf(fmaxf(__half2float(u[c]), -limit), limit);
        float a = gv / (1.0f + __expf(-gv)) * uv;
        s_act[c] = __float2half(a);
    }
    __syncwarp();
    const half* scale = suh + (int64_t) expert_of_row[r] * suh_stride;   // inner routine adds blockIdx.y*128 itself
    had_hf_r_128_inner<true, false>(s_act, act + (int64_t) r * I + blk * 128, scale, HAD_SCALE);
}

// ---------------------------------------------------------------------------------------------- 4./6. grouped GEMM
// A: sorted rows [rows, K] fp16 (already rotated). B: trellis for expert e at packed + e*expert_stride, laid out
// [tiles_k, tiles_n, 64] words; for the gate|up stage two such tensors are addressed through n-tile ranges
// (nt < tiles_n -> gate, else up) with their own svh. Output: fp16 [rows, N] (MODE 0), or fp32 atomicAdd
// scatter into out[token_sorted[r]] scaled by route_weight[r] (MODE 1).
template <int MODE>
__global__ __launch_bounds__(THREADS)
void grouped_gemm_kernel(
    const half* __restrict__ a,
    const uint16_t* __restrict__ packed_gate, const uint16_t* __restrict__ packed_up,   // packed_up may be null
    int64_t expert_stride_words,
    const half* __restrict__ svh_gate, const half* __restrict__ svh_up, int64_t svh_stride,
    const int32_t* __restrict__ block_expert, const int32_t* __restrict__ block_row0,
    const int32_t* __restrict__ counts, const int32_t* __restrict__ expert_row0,
    const int64_t* __restrict__ token_sorted, const half* __restrict__ route_weight,
    void* __restrict__ out, int size_k, int size_n_each, int n_tensors)
{
    const int e = block_expert[blockIdx.y];
    if (e < 0) return;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    half* sh_a = reinterpret_cast<half*>(shared_raw);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(shared_raw + A_BYTES);
    float* sh_c = reinterpret_cast<float*>(shared_raw);

    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int row0 = block_row0[blockIdx.y];                       // absolute sorted-row index of this M block
    const int rows_valid = min(TILE_M, expert_row0[e] + counts[e] - row0);
    const int tiles_n = size_n_each / 16;
    const int n_tile0 = blockIdx.x * N_TILES;                      // in the concatenated [gate|up] tile space
    const int which = (n_tile0 >= tiles_n) ? 1 : 0;                // a 128-col block never straddles (N%128==0)
    const uint16_t* packed = (which ? packed_up : packed_gate) + (int64_t) e * expert_stride_words;
    const half* svh = (which ? svh_up : svh_gate) + (int64_t) e * svh_stride;
    const int nt_local = n_tile0 - which * tiles_n;
    const int n_base_local = nt_local * 16;
    const int num_kb = size_k / TILE_K;

    if (rows_valid < TILE_M)
    {
        const int tail = (TILE_M - rows_valid) * A_ROW_INT4;
        for (int s = 0; s < STAGES; ++s)
        {
            int4* base = reinterpret_cast<int4*>(sh_a + s * A_STAGE_HALFS) + rows_valid * A_ROW_INT4;
            for (int i = t; i < tail; i += THREADS) base[i] = make_int4(0, 0, 0, 0);
        }
        __syncthreads();
    }

    auto load_stage = [&](int kb, int s)
    {
        int4* a_dst = reinterpret_cast<int4*>(sh_a + s * A_STAGE_HALFS);
        const int a_int4 = rows_valid * A_ROW_INT4;
        #pragma unroll
        for (int j = 0; j < (TILE_M * A_ROW_INT4) / THREADS; ++j)
        {
            const int i = t + j * THREADS;
            if (i < a_int4)
            {
                const int row = i / A_ROW_INT4, chunk = i % A_ROW_INT4;
                const half* src = a + (int64_t) (row0 + row) * size_k + kb * TILE_K + chunk * 8;
                cp_async(a_dst + row * A_ROW_INT4 + (chunk ^ (row & (A_ROW_INT4 - 1))), src);
            }
        }
        int4* b_dst = reinterpret_cast<int4*>(sh_b + s * B_STAGE_WORDS);
        constexpr int b_int4 = KT * N_TILES * 8;
        #pragma unroll
        for (int j = 0; j < (b_int4 + THREADS - 1) / THREADS; ++j)
        {
            const int i = t + j * THREADS;
            if (i < b_int4)
            {
                const int kt = i / (N_TILES * 8), nt = (i / 8) % N_TILES, chunk = i & 7;
                const int64_t tile = (int64_t) (kb * KT + kt) * tiles_n + nt_local + nt;
                cp_async(b_dst + (kt * N_TILES + nt) * 8 + chunk, reinterpret_cast<const int4*>(packed + tile * PACKED_WORDS) + chunk);
            }
        }
    };

    FragC frag_c[M_BLOCKS][2];
    #pragma unroll
    for (int mb = 0; mb < M_BLOCKS; ++mb) { frag_c[mb][0] = {}; frag_c[mb][1] = {}; }
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) { if (s < num_kb) load_stage(s, s); cp_async_fence(); }
    for (int kb = 0; kb < num_kb; ++kb)
    {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        { const int nkb = kb + STAGES - 1; if (nkb < num_kb) load_stage(nkb, nkb % STAGES); cp_async_fence(); }
        const int s = kb % STAGES;
        const half* a_stage = sh_a + s * A_STAGE_HALFS;
        const uint16_t* b_stage = sh_b + s * B_STAGE_WORDS;
        #pragma unroll
        for (int kt = 0; kt < KT; ++kt)
        {
            FragB fb0, fb1;
            dq_dispatch<4, 1>(reinterpret_cast<const uint32_t*>(b_stage + (kt * N_TILES + warp) * PACKED_WORDS), lane << 3, fb0, fb1);
            #pragma unroll
            for (int mb = 0; mb < M_BLOCKS; ++mb)
            {
                FragA fa;
                const int row = mb * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
                const int chunk = kt * 2 + (lane >> 4);
                ldsm4(fa, reinterpret_cast<const int4*>(a_stage) + row * A_ROW_INT4 + (chunk ^ (row & (A_ROW_INT4 - 1))));
                ptx_mma_m16n8k16(fa, fb0, frag_c[mb][0]);
                ptx_mma_m16n8k16(fa, fb1, frag_c[mb][1]);
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    const int out_col0 = which * size_n_each + n_base_local;       // column in the concatenated output
    const int out_n = n_tensors * size_n_each;
    #pragma unroll
    for (int mb = 0; mb < M_BLOCKS; ++mb)
    {
        const int rows = min(16, rows_valid - mb * 16);
        if (rows <= 0) break;
        const int r0 = lane >> 2, r1 = r0 + 8, col = (lane & 3) * 2, n0 = warp * 16;
        if (r0 < rows) { float* d = sh_c + r0 * TILE_N + n0 + col; d[0] = frag_c[mb][0][0]; d[1] = frag_c[mb][0][1]; d[8] = frag_c[mb][1][0]; d[9] = frag_c[mb][1][1]; }
        if (r1 < rows) { float* d = sh_c + r1 * TILE_N + n0 + col; d[0] = frag_c[mb][0][2]; d[1] = frag_c[mb][0][3]; d[8] = frag_c[mb][1][2]; d[9] = frag_c[mb][1][3]; }
        __syncthreads();
        for (int row = warp; row < rows; row += WARPS)
            had_ff_128_scale(sh_c + row * TILE_N, sh_c + row * TILE_N, svh + n_base_local);
        __syncthreads();
        for (int i = t; i < rows * TILE_N; i += THREADS)
        {
            const int row = i / TILE_N, c = i % TILE_N;
            const int64_t srow = row0 + mb * 16 + row;
            const float v = sh_c[i];
            if constexpr (MODE == 0)
                reinterpret_cast<half*>(out)[srow * out_n + out_col0 + c] = __float2half(v);
            else
                atomicAdd(reinterpret_cast<float*>(out) + token_sorted[srow] * out_n + out_col0 + c, v * __half2float(route_weight[srow]));
        }
        __syncthreads();
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------- host API
// block table: counts int32[E] -> block_expert/block_row0 int32[max_blocks], meta int32[2] = {n_blocks, n_rows}
void build_blocks(at::Tensor counts, at::Tensor block_expert, at::Tensor block_row0, at::Tensor meta)
{
    const at::cuda::OptionalCUDAGuard g(counts.device());
    TORCH_CHECK(counts.scalar_type() == at::kInt && block_expert.scalar_type() == at::kInt && block_row0.scalar_type() == at::kInt && meta.scalar_type() == at::kInt, "int32 tensors");
    TORCH_CHECK(counts.numel() <= MAX_EXPERTS, "too many experts");
    build_blocks_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        counts.data_ptr<int32_t>(), (int) counts.numel(), (int) block_expert.numel(),
        block_expert.data_ptr<int32_t>(), block_row0.data_ptr<int32_t>(), meta.data_ptr<int32_t>());
    cuda_check(cudaPeekAtLastError());
}

// dst[r] = H128( src[token_sorted[r] or r] ⊙ suh[expert_of_row[r]] )
void row_had(at::Tensor src, at::Tensor dst, at::Tensor token_sorted, at::Tensor expert_of_row, at::Tensor suh, bool gather)
{
    const at::cuda::OptionalCUDAGuard g(src.device());
    const int rows = (int) dst.size(0), K = (int) dst.size(1);
    TORCH_CHECK(K % 128 == 0 && suh.size(-1) == K, "K must be a multiple of 128 and match suh");
    dim3 grid(rows, K / 128), block(32);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (gather)
        row_had_kernel<true><<<grid, block, 0, stream>>>((const half*) src.data_ptr(), (half*) dst.data_ptr(), token_sorted.data_ptr<int64_t>(),
            expert_of_row.data_ptr<int32_t>(), (const half*) suh.data_ptr(), suh.stride(0), rows, K);
    else
        row_had_kernel<false><<<grid, block, 0, stream>>>((const half*) src.data_ptr(), (half*) dst.data_ptr(), nullptr,
            expert_of_row.data_ptr<int32_t>(), (const half*) suh.data_ptr(), suh.stride(0), rows, K);
    cuda_check(cudaPeekAtLastError());
}

void act_had(at::Tensor gu, at::Tensor act, at::Tensor expert_of_row, at::Tensor suh_down, double limit)
{
    const at::cuda::OptionalCUDAGuard g(gu.device());
    const int rows = (int) act.size(0), I = (int) act.size(1);
    TORCH_CHECK(gu.size(1) == 2 * I && I % 128 == 0, "gate|up width must be 2I, I multiple of 128");
    dim3 grid(rows, I / 128), block(32);
    act_had_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream().stream()>>>((const half*) gu.data_ptr(), (half*) act.data_ptr(),
        expert_of_row.data_ptr<int32_t>(), (const half*) suh_down.data_ptr(), suh_down.stride(0), rows, I, (float) limit);
    cuda_check(cudaPeekAtLastError());
}

// gate|up: packed_gate/packed_up [E, tiles_k, tiles_n, 64] int16 (same strides), svh [E, N] each; out fp16 [rows, 2N]
// down:    packed_gate = w2 [E, tiles_k, tiles_n, 64], packed_up undefined/none; out fp32 [T, N] atomically scattered
void grouped_gemm(at::Tensor a, at::Tensor packed_gate, c10::optional<at::Tensor> packed_up,
                  at::Tensor svh_gate, c10::optional<at::Tensor> svh_up,
                  at::Tensor block_expert, at::Tensor block_row0, at::Tensor counts, at::Tensor expert_row0,
                  at::Tensor token_sorted, at::Tensor route_weight, at::Tensor out, int64_t mode)
{
    const at::cuda::OptionalCUDAGuard g(a.device());
    TORCH_CHECK(a.scalar_type() == at::kHalf && a.is_contiguous(), "a fp16 contiguous");
    TORCH_CHECK(packed_gate.dim() == 4 && packed_gate.size(3) == PACKED_WORDS && packed_gate.stride(3) == 1
                && packed_gate.stride(2) == PACKED_WORDS && packed_gate.stride(1) == packed_gate.size(2) * PACKED_WORDS,
                "packed [E, tiles_k, tiles_n, 64] with contiguous inner dims (expert stride may be larger: [E,2,...] views)");
    const int size_k = (int) a.size(1), tiles_k = (int) packed_gate.size(1), tiles_n = (int) packed_gate.size(2);
    TORCH_CHECK(size_k == tiles_k * 16 && size_k % TILE_K == 0, "K mismatch");
    const int size_n_each = tiles_n * 16;
    TORCH_CHECK(size_n_each % TILE_N == 0, "N must be a multiple of 128");
    const int n_tensors = packed_up.has_value() ? 2 : 1;
    const int64_t expert_stride = packed_gate.stride(0);
    const int64_t svh_stride = svh_gate.stride(0);
    TORCH_CHECK(svh_gate.size(-1) == size_n_each, "svh N mismatch");
    if (packed_up.has_value())
    {
        TORCH_CHECK(packed_up->stride(0) == expert_stride && svh_up.has_value() && svh_up->stride(0) == svh_stride, "gate/up strides must match");
    }
    const int max_blocks = (int) block_expert.numel();
    dim3 grid(n_tensors * size_n_each / TILE_N, max_blocks), block(THREADS);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    static bool attr_set = false;
    if (!attr_set)
    {
        cudaFuncSetAttribute(grouped_gemm_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
        cudaFuncSetAttribute(grouped_gemm_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
        attr_set = true;
    }
    const uint16_t* pg = (const uint16_t*) packed_gate.data_ptr();
    const uint16_t* pu = packed_up.has_value() ? (const uint16_t*) packed_up->data_ptr() : nullptr;
    const half* sg = (const half*) svh_gate.data_ptr();
    const half* su = svh_up.has_value() ? (const half*) svh_up->data_ptr() : nullptr;
    if (mode == 0)
    {
        TORCH_CHECK(out.scalar_type() == at::kHalf && out.size(1) == n_tensors * size_n_each, "mode 0: out fp16 [rows, n_tensors*N]");
        grouped_gemm_kernel<0><<<grid, block, SMEM_TOTAL, stream>>>((const half*) a.data_ptr(), pg, pu, expert_stride, sg, su, svh_stride,
            block_expert.data_ptr<int32_t>(), block_row0.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), expert_row0.data_ptr<int32_t>(),
            nullptr, nullptr, out.data_ptr(), size_k, size_n_each, n_tensors);
    }
    else
    {
        TORCH_CHECK(out.scalar_type() == at::kFloat && out.size(1) == size_n_each && n_tensors == 1, "mode 1: out fp32 [T, N], single tensor");
        TORCH_CHECK(token_sorted.scalar_type() == at::kLong && route_weight.scalar_type() == at::kHalf, "routing dtypes");
        grouped_gemm_kernel<1><<<grid, block, SMEM_TOTAL, stream>>>((const half*) a.data_ptr(), pg, nullptr, expert_stride, sg, nullptr, svh_stride,
            block_expert.data_ptr<int32_t>(), block_row0.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), expert_row0.data_ptr<int32_t>(),
            token_sorted.data_ptr<int64_t>(), (const half*) route_weight.data_ptr(), out.data_ptr(), size_k, size_n_each, n_tensors);
    }
    cuda_check(cudaPeekAtLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("build_blocks", &build_blocks, "per-expert M-block table from routing counts");
    m.def("row_had", &row_had, "per-row H128 with per-expert sign vector (optional gather)");
    m.def("act_had", &act_had, "GLM swiglu (clamped) + H128 with per-expert down sign vector");
    m.def("grouped_gemm", &grouped_gemm, "grouped weight-stationary trellis GEMM over all experts");
}
