// exl3_fat_gemm_v2 — design 1: weight-stationary, pipelined trellis GEMM for fat experts.
//
// Same contract as PR77's exl3_fat_gemm / exl3_fat_gemm_scatter (K4 MCG trellis, fp16 rotated
// input rows, fp32 output with the H128 * svh epilogue), same codebook decode (dq_dispatch<4,1>),
// so the result is bit-comparable to the E2 kernel. Only the mainloop changes:
//   * K stage of TILE_K (32 or 64) columns instead of 16, so each barrier pair covers KT trellis
//     tiles per warp: KT decodes + KT*8*2 MMAs, instead of 1 decode + 16 MMAs.
//   * STAGES-deep cp.async pipeline: the next K stage of A (activations) and packed B (trellis)
//     lands in shared memory while the current one is being multiplied.
//   * A tile in shared memory is XOR-swizzled by (row & (chunks_per_row-1)) so ldmatrix.x4 is bank-conflict
//     free at 8 int4 chunks per row (K64); 2-way at K32.
//   * The epilogue staging buffer aliases the A stage buffers after the mainloop.
// Rows beyond size_m in the last M block are zero in shared memory (zeroed once; never written).
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
constexpr int M_BLOCKS = TILE_M / 16;      // 8
constexpr int N_TILES = TILE_N / 16;       // 8, one trellis tile column per warp
constexpr int PACKED_WORDS = 4 * 16;       // uint16 per 16x16 K4 tile (128 bytes)
constexpr float HAD_SCALE = 0.088388347648f;

__device__ inline void v2_had_ff_128(const float* input_ptr, float* output_ptr, const half* scale)
{
    int lane = threadIdx.x & 31;
    float4 v = reinterpret_cast<const float4*>(input_ptr)[lane];
    float s0 = v.x + v.y, d0 = v.x - v.y, s1 = v.z + v.w, d1 = v.z - v.w;
    v.x = s0 + s1; v.y = d0 + d1; v.z = s0 - s1; v.w = d0 - d1;
    shuffle_had_f2x32(v.x, v.y, lane);
    shuffle_had_f2x32(v.z, v.w, lane);
    v.x *= HAD_SCALE; v.y *= HAD_SCALE; v.z *= HAD_SCALE; v.w *= HAD_SCALE;
    half4 scales = reinterpret_cast<const half4*>(scale)[lane];
    v.x *= __low2float(scales.x); v.y *= __high2float(scales.x);
    v.z *= __low2float(scales.y); v.w *= __high2float(scales.y);
    reinterpret_cast<float4*>(output_ptr)[lane] = v;
}

template <int TILE_K, int STAGES>
struct Smem
{
    static constexpr int KT = TILE_K / 16;
    static constexpr int A_STAGE_HALFS = TILE_M * TILE_K;
    static constexpr int A_ROW_INT4 = TILE_K / 8;
    static constexpr int B_STAGE_WORDS = KT * N_TILES * PACKED_WORDS;
    static constexpr int A_BYTES = STAGES * A_STAGE_HALFS * (int) sizeof(half);
    static constexpr int B_BYTES = STAGES * B_STAGE_WORDS * (int) sizeof(uint16_t);
    static constexpr int C_BYTES = 16 * TILE_N * (int) sizeof(float);
    static constexpr int TOTAL = A_BYTES + B_BYTES;            // C aliases A after the mainloop
    static_assert(C_BYTES <= A_BYTES, "epilogue buffer must fit in the A stages");
    static_assert(TILE_K % 16 == 0 && TILE_K >= 16, "TILE_K must be a multiple of 16");
};

// One template parameter only: the CUDA 13 launch-stub macro splits multi-parameter template
// argument lists at the commas. V = variant * 2 + scatter.
constexpr int cfg_tile_k(int V)  { const int v = V >> 1; return (v < 2) ? 64 : 32; }
constexpr int cfg_stages(int V)  { const int v = V >> 1; return v == 0 ? 3 : v == 1 ? 2 : v == 2 ? 4 : 3; }
constexpr bool cfg_scatter(int V) { return (V & 1) != 0; }

template <int V>
__global__ __launch_bounds__(THREADS)
void exl3_fat_gemm_v2_kernel(
    const half* __restrict__ a,
    const uint16_t* __restrict__ packed,
    float* __restrict__ out,
    const half* __restrict__ svh,
    const int64_t* __restrict__ token_idx,
    const half* __restrict__ route_weight,
    int size_m, int size_k, int size_n)
{
    constexpr int TILE_K = cfg_tile_k(V);
    constexpr int STAGES = cfg_stages(V);
    constexpr bool scatter = cfg_scatter(V);
    using S = Smem<TILE_K, STAGES>;
    constexpr int KT = S::KT;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    half* sh_a = reinterpret_cast<half*>(shared_raw);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(shared_raw + S::A_BYTES);
    float* sh_c = reinterpret_cast<float*>(shared_raw);   // aliases the A stages after the loop

    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const int m_base = blockIdx.y * TILE_M;
    const int n_base = blockIdx.x * TILE_N;
    const int tiles_n = size_n / 16;
    const int rows_valid = min(TILE_M, size_m - m_base);
    const int num_kb = size_k / TILE_K;

    // Zero the rows past size_m in every A stage once; the loaders never touch them.
    if (rows_valid < TILE_M)
    {
        const int tail_int4 = (TILE_M - rows_valid) * S::A_ROW_INT4;
        for (int s = 0; s < STAGES; ++s)
        {
            int4* base = reinterpret_cast<int4*>(sh_a + s * S::A_STAGE_HALFS) + rows_valid * S::A_ROW_INT4;
            for (int i = t; i < tail_int4; i += THREADS) base[i] = make_int4(0, 0, 0, 0);
        }
        __syncthreads();
    }

    auto load_stage = [&](int kb, int s)
    {
        // A: rows_valid x TILE_K halves, 16-byte chunks, swizzled chunk' = chunk ^ (row & 7)
        int4* a_dst = reinterpret_cast<int4*>(sh_a + s * S::A_STAGE_HALFS);
        const int a_int4 = rows_valid * S::A_ROW_INT4;
        #pragma unroll
        for (int j = 0; j < (TILE_M * S::A_ROW_INT4) / THREADS; ++j)
        {
            const int i = t + j * THREADS;
            if (i < a_int4)
            {
                const int row = i / S::A_ROW_INT4;
                const int chunk = i % S::A_ROW_INT4;
                const half* src = a + (int64_t) (m_base + row) * size_k + kb * TILE_K + chunk * 8;
                cp_async(a_dst + row * S::A_ROW_INT4 + (chunk ^ (row & (S::A_ROW_INT4 - 1))), src);
            }
        }
        // B: KT k-tiles x N_TILES n-tiles x 128 bytes; one int4 per thread per pass
        int4* b_dst = reinterpret_cast<int4*>(sh_b + s * S::B_STAGE_WORDS);
        constexpr int b_int4 = KT * N_TILES * 8;
        #pragma unroll
        for (int j = 0; j < (b_int4 + THREADS - 1) / THREADS; ++j)
        {
            const int i = t + j * THREADS;
            if (i < b_int4)
            {
                const int kt = i / (N_TILES * 8);
                const int nt = (i / 8) % N_TILES;
                const int chunk = i & 7;
                const int64_t tile = (int64_t) (kb * KT + kt) * tiles_n + (n_base / 16) + nt;
                const int4* src = reinterpret_cast<const int4*>(packed + tile * PACKED_WORDS) + chunk;
                cp_async(b_dst + (kt * N_TILES + nt) * 8 + chunk, src);
            }
        }
    };

    FragC frag_c[M_BLOCKS][2];
    #pragma unroll
    for (int mb = 0; mb < M_BLOCKS; ++mb) { frag_c[mb][0] = {}; frag_c[mb][1] = {}; }

    // Prologue: stages 0 .. STAGES-2 in flight
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (s < num_kb) load_stage(s, s);
        cp_async_fence();
    }

    for (int kb = 0; kb < num_kb; ++kb)
    {
        cp_async_wait<STAGES - 2>();
        __syncthreads();                       // stage kb visible; everyone done with stage kb-1
        {
            const int nkb = kb + STAGES - 1;
            if (nkb < num_kb) load_stage(nkb, nkb % STAGES);
            cp_async_fence();
        }
        const int s = kb % STAGES;
        const half* a_stage = sh_a + s * S::A_STAGE_HALFS;
        const uint16_t* b_stage = sh_b + s * S::B_STAGE_WORDS;
        #pragma unroll
        for (int kt = 0; kt < KT; ++kt)
        {
            FragB frag_b0, frag_b1;
            const uint32_t* warp_b = reinterpret_cast<const uint32_t*>(b_stage + (kt * N_TILES + warp) * PACKED_WORDS);
            dq_dispatch<4, 1>(warp_b, lane << 3, frag_b0, frag_b1);
            #pragma unroll
            for (int mb = 0; mb < M_BLOCKS; ++mb)
            {
                FragA frag_a;
                const int row = mb * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
                const int chunk = kt * 2 + (lane >> 4);
                ldsm4(frag_a, reinterpret_cast<const int4*>(a_stage) + row * S::A_ROW_INT4 + (chunk ^ (row & (S::A_ROW_INT4 - 1))));
                ptx_mma_m16n8k16(frag_a, frag_b0, frag_c[mb][0]);
                ptx_mma_m16n8k16(frag_a, frag_b1, frag_c[mb][1]);
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();                           // all warps out of the A stages before sh_c aliases them

    // Epilogue: identical to E2 — stage 16 rows, H128 + svh per row, store or scatter.
    #pragma unroll
    for (int mb = 0; mb < M_BLOCKS; ++mb)
    {
        const int rows = min(16, rows_valid - mb * 16);
        if (rows <= 0) break;
        const int row0 = lane >> 2, row1 = row0 + 8, col = (lane & 3) * 2, n0 = warp * 16;
        if (row0 < rows)
        {
            float* d = sh_c + row0 * TILE_N + n0 + col;
            d[0] = frag_c[mb][0][0]; d[1] = frag_c[mb][0][1]; d[8] = frag_c[mb][1][0]; d[9] = frag_c[mb][1][1];
        }
        if (row1 < rows)
        {
            float* d = sh_c + row1 * TILE_N + n0 + col;
            d[0] = frag_c[mb][0][2]; d[1] = frag_c[mb][0][3]; d[8] = frag_c[mb][1][2]; d[9] = frag_c[mb][1][3];
        }
        __syncthreads();
        for (int row = warp; row < rows; row += WARPS)
            v2_had_ff_128(sh_c + row * TILE_N, sh_c + row * TILE_N, svh + n_base);
        __syncthreads();
        for (int i = t; i < rows * TILE_N; i += THREADS)
        {
            const int row = i / TILE_N, col_out = i % TILE_N;
            const int source_row = m_base + mb * 16 + row;
            float value = sh_c[i];
            if constexpr (scatter)
            {
                const int64_t destination = token_idx[source_row];
                value *= __half2float(route_weight[source_row]);
                out[destination * size_n + n_base + col_out] += value;   // one route per token per expert
            }
            else
            {
                out[(int64_t) source_row * size_n + n_base + col_out] = value;
            }
        }
        __syncthreads();
    }
}

void check_common(const at::Tensor& a, const at::Tensor& packed, const at::Tensor& out, const at::Tensor& svh,
                  int64_t K, bool mcg, bool mul1)
{
    TORCH_CHECK(a.is_cuda() && packed.is_cuda() && out.is_cuda() && svh.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(a.is_contiguous() && packed.is_contiguous() && out.is_contiguous() && svh.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(a.scalar_type() == at::kHalf, "a must be float16");
    TORCH_CHECK(packed.scalar_type() == at::kShort, "packed must be int16");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
    TORCH_CHECK(svh.scalar_type() == at::kHalf, "svh must be float16");
    TORCH_CHECK(a.dim() == 2 && packed.dim() == 3 && out.dim() == 2 && svh.dim() == 1, "rank mismatch");
    TORCH_CHECK(K == 4 && mcg && !mul1, "only K4 MCG tensors");
    TORCH_CHECK(a.size(1) == packed.size(0) * 16, "a K dimension does not match packed tensor");
    TORCH_CHECK(svh.numel() == packed.size(1) * 16, "svh N dimension does not match packed tensor");
    TORCH_CHECK(svh.numel() % TILE_N == 0, "N must be divisible by 128");
    TORCH_CHECK(packed.size(2) == PACKED_WORDS, "packed K4 block width must be 64 int16 words");
    TORCH_CHECK(a.size(1) % 64 == 0, "K must be divisible by 64");
}

template <int V>
void launch_variant(const at::Tensor& a, const at::Tensor& packed, at::Tensor& out, const at::Tensor& svh,
                    const at::Tensor& token_idx, const at::Tensor& route_weight)
{
    constexpr bool scatter = cfg_scatter(V);
    using S = Smem<cfg_tile_k(V), cfg_stages(V)>;
    auto kernel = exl3_fat_gemm_v2_kernel<V>;
    static bool attr_set = false;
    if (!attr_set)
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, S::TOTAL);
        attr_set = true;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int size_m = (int) a.size(0), size_k = (int) a.size(1), size_n = (int) svh.numel();
    dim3 block(THREADS);
    dim3 grid(size_n / TILE_N, (size_m + TILE_M - 1) / TILE_M);
    kernel<<<grid, block, S::TOTAL, stream>>>(
        reinterpret_cast<const half*>(a.data_ptr()), reinterpret_cast<const uint16_t*>(packed.data_ptr()),
        reinterpret_cast<float*>(out.data_ptr()), reinterpret_cast<const half*>(svh.data_ptr()),
        scatter ? reinterpret_cast<const int64_t*>(token_idx.data_ptr()) : nullptr,
        scatter ? reinterpret_cast<const half*>(route_weight.data_ptr()) : nullptr,
        size_m, size_k, size_n);
    cuda_check(cudaPeekAtLastError());
}

template <bool scatter>
void dispatch(int variant, const at::Tensor& a, const at::Tensor& packed, at::Tensor& out, const at::Tensor& svh,
              const at::Tensor& token_idx, const at::Tensor& route_weight)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    constexpr int sc = scatter ? 1 : 0;
    switch (variant)
    {
        case 0: launch_variant<0 * 2 + sc>(a, packed, out, svh, token_idx, route_weight); break;   // K64, 3 stages: 60 KB
        case 1: launch_variant<1 * 2 + sc>(a, packed, out, svh, token_idx, route_weight); break;   // K64, 2 stages: 40 KB
        case 2: launch_variant<2 * 2 + sc>(a, packed, out, svh, token_idx, route_weight); break;   // K32, 4 stages: 40 KB
        case 3: launch_variant<3 * 2 + sc>(a, packed, out, svh, token_idx, route_weight); break;   // K32, 3 stages: 30 KB
        default: TORCH_CHECK(false, "unknown variant");
    }
}

}  // namespace

void fat_gemm_v2(at::Tensor a, at::Tensor packed, at::Tensor out, at::Tensor svh, int64_t K, bool mcg, bool mul1, int64_t variant)
{
    check_common(a, packed, out, svh, K, mcg, mul1);
    TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == svh.numel(), "out shape must be [M, N]");
    dispatch<false>((int) variant, a, packed, out, svh, at::Tensor(), at::Tensor());
}

void fat_gemm_v2_scatter(at::Tensor a, at::Tensor packed, at::Tensor out, at::Tensor svh, at::Tensor token_idx,
                         at::Tensor route_weight, int64_t K, bool mcg, bool mul1, int64_t variant)
{
    check_common(a, packed, out, svh, K, mcg, mul1);
    TORCH_CHECK(out.size(1) == svh.numel(), "out N dimension mismatch");
    TORCH_CHECK(token_idx.is_cuda() && route_weight.is_cuda() && token_idx.is_contiguous() && route_weight.is_contiguous(), "routing tensors");
    TORCH_CHECK(token_idx.scalar_type() == at::kLong && route_weight.scalar_type() == at::kHalf, "routing dtypes");
    TORCH_CHECK(token_idx.numel() == a.size(0) && route_weight.numel() == a.size(0), "routing tensors must have M elements");
    dispatch<true>((int) variant, a, packed, out, svh, token_idx, route_weight);
}

int64_t smem_bytes(int64_t variant)
{
    switch (variant)
    {
        case 0: return Smem<64, 3>::TOTAL;
        case 1: return Smem<64, 2>::TOTAL;
        case 2: return Smem<32, 4>::TOTAL;
        case 3: return Smem<32, 3>::TOTAL;
        default: return -1;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fat_gemm_v2", &fat_gemm_v2, "weight-stationary pipelined trellis GEMM (design 1)");
    m.def("fat_gemm_v2_scatter", &fat_gemm_v2_scatter, "scatter variant");
    m.def("smem_bytes", &smem_bytes, "dynamic shared memory per variant");
}
