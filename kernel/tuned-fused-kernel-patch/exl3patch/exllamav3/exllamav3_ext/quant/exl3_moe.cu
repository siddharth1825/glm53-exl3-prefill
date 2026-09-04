#include <cuda_fp16.h>
#include "exl3_gemm.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "comp_units/exl3_moe_instances.cuh"
#include "exl3_devctx.cuh"
#include <set>
#include <cstdlib>
#include <cstring>

int exl3_moe_max_concurrency(int device)
{
    int num_sms = DevCtx::instance().get_num_sms(device);
    // EXL3_MOE_OCC2=1: two blocks per SM -> two expert groups per 8 SMs
    const char* e = getenv("EXL3_MOE_OCC2"); int mult = (e && atoi(e)) ? 2 : 1;
    return mult * num_sms / MOE_SMS_PER_EXPERT;
}

std::set<void*> moe_kernel_attr_set[MAX_DEVICES] = {};

fp_exl3_moe_kernel exl3_moe_kernel_instances[] =
{
    exl3_moe_kernel_k0_n128(), exl3_moe_kernel_k0_n256(), // Switch Kg, Ku and Kd at runtime
    exl3_moe_kernel_k1_n128(), exl3_moe_kernel_k1_n256(), // Compile-time Kg = Ku = Kd
    exl3_moe_kernel_k2_n128(), exl3_moe_kernel_k2_n256(), // ...
    exl3_moe_kernel_k3_n128(), exl3_moe_kernel_k3_n256(),
    exl3_moe_kernel_k4_n128(), exl3_moe_kernel_k4_n256(),
    exl3_moe_kernel_k5_n128(), exl3_moe_kernel_k5_n256(),
    exl3_moe_kernel_k6_n128(), exl3_moe_kernel_k6_n256(),
    exl3_moe_kernel_k7_n128(), exl3_moe_kernel_k7_n256(),
    exl3_moe_kernel_k8_n128(), exl3_moe_kernel_k8_n256()
};

// Prefill instances: 64 rows per GEMM pass so each dequantised weight fragment is
// reused across four M blocks instead of being re-read and re-decoded per 16 rows.
// Selected when the batch is larger than one decode tile. Env EXL3_MOE_PREFILL_M=16
// forces the original path (A/B switch for validation).
static fp_exl3_moe_kernel exl3_moe_prefill_kernel(int K, int N_off)
{
    // EXL3_MOE_PREFILL_M: 16 (original path) | 32 | 64 (default 64). EXL3_MOE_PREFILL_N: 128 | 256
    // (default: same tile N the decode kernel would pick for these dims).
    static int M = -1, N = -1;
    if (M < 0) { const char* e = getenv("EXL3_MOE_PREFILL_M"); M = e ? atoi(e) : 64; }
    if (N < 0) { const char* e = getenv("EXL3_MOE_PREFILL_N"); N = e ? atoi(e) : 0; }
    int n = N ? N : (N_off ? 256 : 128);
    static int occ2 = -1;
    if (occ2 < 0) { const char* e = getenv("EXL3_MOE_OCC2"); occ2 = (e && atoi(e)) ? 1 : 0; }
    static const char* st = getenv("EXL3_MOE_STAGES");
    if (st && K == 4 && n == 256 && M == 16)
    {
        if (!strcmp(st, "s3f2")) return exl3_moe_kernel_k4_n256_m16_s3f2();
        if (!strcmp(st, "s4f3")) return exl3_moe_kernel_k4_n256_m16_s4f3();
        if (!strcmp(st, "s6f3")) return exl3_moe_kernel_k4_n256_m16_s6f3();
        if (!strcmp(st, "s6f2")) return exl3_moe_kernel_k4_n256_m16_s6f2();
        if (!strcmp(st, "s8f2")) return exl3_moe_kernel_k4_n256_m16_s8f2();
    }
    if (occ2 && K == 4 && n == 256 && M == 16) return exl3_moe_kernel_k4_n256_m16_occ2();
    if (occ2 && K == 4 && n == 256 && M == 32) return exl3_moe_kernel_k4_n256_m32_occ2();
    if (M == 16) return nullptr;
    if (K == 4 && n == 128 && M == 32) return exl3_moe_kernel_k4_n128_m32();
    if (K == 4 && n == 128 && M == 64) return exl3_moe_kernel_k4_n128_m64();
    if (K == 4 && n == 256 && M == 32) return exl3_moe_kernel_k4_n256_m32();
    if (K == 4 && n == 256 && M == 64) return exl3_moe_kernel_k4_n256_m64();
    if (K == 0 && n == 128 && M == 32) return exl3_moe_kernel_k0_n128_m32();
    if (K == 0 && n == 128 && M == 64) return exl3_moe_kernel_k0_n128_m64();
    // (runtime-K at N=256 does not fit shared memory when M-tiled; falls back to the original path)
    return nullptr;
}

/*
Fused mixture-of-experts MLP operation for EXL3 weights

inputs:
    hidden_state:
        input hidden state - shape (bsz, hidden_dim) - fp16

    output_state:
        output hidden state - shape (bsz, hidden_dim) - fp32
        zero-initialized

    expert_count:
        bincount of expert indices across all tokens in batch - shape (num_experts + 1,) - int64
        last item is ignored, used for the case where some tokens may activate less than num_experts_per_token
        experts (specifically in expert split mode)

    token_sorted:
        token indices, sorted by expert - shape (bsz * num_experts_per_tok,)  - int64

    weight_sorted:
        routing weight per token, sorted by expert - shape (bsz * num_experts_per_tok,) - fp16

    temp_state_g:
    temp_state_u:
        temp state storage - shape (concurrency, max_tokens_per_expert, hidden_dim), fp16

    temp_intermediate_g
    temp_intermediate_u:
        temp intermediate storage - shape (concurrency, max_tokens_per_expert, intermediate_dim), fp16

    act_function:
        int, see exl3_moe.cuh

    K_gate
    K_up
    K_down:
        int, bitrates for gate, up, down tensors

    gate_ptrs_trellis
    gate_ptrs_suh
    gate_ptrs_svh
    up_ptrs_trellis
    up_ptrs_suh
    up_ptrs_svh
    down_ptrs_trellis
    down_ptrs_suh
    down_ptrs_svh:
        tensors of data_ptrs to quantized tensor data - each shape (num_experts,) - void*

    gate_mcg
    gate_mul1
    up_mcg
    up_mul1
    down_mcg
    down_mul1:
        bool, codebook flags
*/

void exl3_moe
(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_count,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,

    const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u,

    const int act_function,

    const int K_gate,
    const int K_up,
    const int K_down,

    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,

    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,

    const float act_limit
)
{
    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Validate args
    TORCH_CHECK_DTYPE(hidden_state, kHalf);
    TORCH_CHECK_DIM(hidden_state, 2);
    size_t bsz = hidden_state.size(0);
    size_t hidden_dim = hidden_state.size(1);

    TORCH_CHECK_DTYPE(output_state, kFloat);
    TORCH_CHECK_SHAPES_FULL(output_state, hidden_state);

    TORCH_CHECK_DTYPE(expert_count, kLong);
    TORCH_CHECK_DIM(expert_count, 1);
    size_t num_experts = expert_count.size(0) - 1;

    TORCH_CHECK_DTYPE(token_sorted, kLong);
    TORCH_CHECK_DIM(token_sorted, 1);
    TORCH_CHECK_SHAPES_FULL(token_sorted, weight_sorted);
    size_t num_experts_per_tok = token_sorted.size(0) / bsz;

    TORCH_CHECK_DTYPE(temp_state_g, kHalf);
    TORCH_CHECK_DTYPE(temp_state_u, kHalf);
    TORCH_CHECK_DIM(temp_state_g, 3);
    TORCH_CHECK_SHAPES(temp_state_g, 2, hidden_state, 1, 1);
    TORCH_CHECK_SHAPES_FULL(temp_state_g, temp_state_u);
    size_t max_tokens_per_expert = temp_state_g.size(1);
    size_t concurrency = temp_state_g.size(0);

    TORCH_CHECK_DTYPE(temp_intermediate_g, kHalf);
    TORCH_CHECK_DTYPE(temp_intermediate_u, kHalf);
    TORCH_CHECK_DIM(temp_intermediate_g, 3);
    TORCH_CHECK_DIM(temp_intermediate_u, 3);
    TORCH_CHECK_SHAPES_FULL(temp_intermediate_g, temp_intermediate_u);
    TORCH_CHECK_SHAPES(temp_intermediate_g, 1, temp_state_g, 1, 1);
    size_t intermediate_dim = temp_intermediate_g.size(2);

    // TORCH_CHECK(!(gate_mcg && gate_mul1), "Specified both mcg and mul1 (gate)");
    // TORCH_CHECK(!(up_mcg && up_mul1), "Specified both mcg and mul1 (up)");
    // TORCH_CHECK(!(down_mcg && down_mul1), "Specified both mcg and mul1 (down)");
    TORCH_CHECK(gate_mcg && !gate_mul1, "MoE kernel: Only mcg codebook is currently supported");
    TORCH_CHECK(up_mcg && !up_mul1, "MoE kernel: Only mcg codebook is currently supported");
    TORCH_CHECK(down_mcg && !down_mul1, "MoE kernel: Only mcg codebook is currently supported");

    // TORCH_CHECK(act_function == MOE_ACT_SILU, "MoE kernel: Only SiLU is currently supported");

    int K = 0;
    if (K_gate == K_up && K_up == K_down) K = K_gate;

    TORCH_CHECK_DIM(gate_ptrs_trellis, 1);
    TORCH_CHECK(gate_ptrs_trellis.size(0) == num_experts, "Number of gate tensors doesn't match num_experts");
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, gate_ptrs_suh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, gate_ptrs_svh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, up_ptrs_trellis);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, up_ptrs_suh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, up_ptrs_svh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, down_ptrs_trellis);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, down_ptrs_suh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, down_ptrs_svh);

    // Device properties
    int device;
    cudaGetDevice(&device);
    int num_sms = DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    // Launch
    int block_dim = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;
    TORCH_CHECK(concurrency * MOE_SMS_PER_EXPERT <= 2 * num_sms, "Concurrency too high for device num_sms");
    dim3 grid_dim(MOE_SMS_PER_EXPERT, 1, concurrency);

    int N_off = 0;
    if (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0) N_off = 1;
    fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[2 * K + N_off];
    if (bsz > 16)
    {
        fp_exl3_moe_kernel pk = exl3_moe_prefill_kernel(K, N_off);
        if (pk) kernel = pk;
    }
    // Dynamic smem: the launcher used to request SMEM_MAX for every block, which alone caps
    // occupancy at one block per SM. EXL3_MOE_SMEM_FIT=1 requests the shape's real need.
    static int smem_fit = -1;
    if (smem_fit < 0) { const char* e = getenv("EXL3_MOE_SMEM_FIT"); smem_fit = (e && atoi(e)) ? 1 : 0; }
    int smem = SMEM_MAX;
    if (smem_fit)
    {
        static int pm = -1; if (pm < 0) { const char* e = getenv("EXL3_MOE_PREFILL_M"); pm = e ? atoi(e) : 64; }
        int tm = (bsz > 16 && kernel != exl3_moe_kernel_instances[2 * K + N_off]) ? pm : 16;
        int tn = N_off ? 256 : 128;
        int kb = K ? K : 8;
        int sh_a = tm * MOE_TILESIZE_K * 2;                                   // halfs -> bytes
        int sh_b = (MOE_TILESIZE_K / 16) * (tn / 16) * 256 / 16 * kb * 2;      // uint16 -> bytes
        int frags_n = 2 * (tn / 16) / (EXL3_GEMM_BASE_THREADS / 32);
        int sh_c = 4 * EXL3_GEMM_BASE_THREADS * frags_n * (tm / 16) * 4;       // floats -> bytes
        int sh_stages = MOE_SH_STAGES;
        { const char* st = getenv("EXL3_MOE_STAGES"); if (st && st[0] == 's') sh_stages = atoi(st + 1); }
        smem = sh_stages * (sh_a + sh_b) + sh_c;
    }

    if (moe_kernel_attr_set[device].find((void*) kernel) == moe_kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
        moe_kernel_attr_set[device].insert((void*) kernel);
        cuda_check(cudaPeekAtLastError());
    }

    void* _hidden_state = hidden_state.data_ptr();
    void* _temp_state_g = temp_state_g.data_ptr();
    void* _temp_state_u = temp_state_u.data_ptr();
    void* _temp_intermediate_g = temp_intermediate_g.data_ptr();
    void* _temp_intermediate_u = temp_intermediate_u.data_ptr();
    void* _output_state = output_state.data_ptr();

    void* _gate_ptrs_trellis = gate_ptrs_trellis.data_ptr();
    void* _gate_ptrs_suh = gate_ptrs_suh.data_ptr();
    void* _gate_ptrs_svh = gate_ptrs_svh.data_ptr();
    void* _up_ptrs_trellis = up_ptrs_trellis.data_ptr();
    void* _up_ptrs_suh = up_ptrs_suh.data_ptr();
    void* _up_ptrs_svh = up_ptrs_svh.data_ptr();
    void* _down_ptrs_trellis = down_ptrs_trellis.data_ptr();
    void* _down_ptrs_suh = down_ptrs_suh.data_ptr();
    void* _down_ptrs_svh = down_ptrs_svh.data_ptr();

    void* _expert_count = expert_count.data_ptr();
    void* _token_sorted = token_sorted.data_ptr();
    void* _weight_sorted = weight_sorted.data_ptr();

    void* kernelArgs[] =
    {
        &_hidden_state,
        &_temp_state_g,
        &_temp_state_u,
        &_temp_intermediate_g,
        &_temp_intermediate_u,
        &_output_state,
        &_gate_ptrs_trellis,
        &_gate_ptrs_suh,
        &_gate_ptrs_svh,
        &_up_ptrs_trellis,
        &_up_ptrs_suh,
        &_up_ptrs_svh,
        &_down_ptrs_trellis,
        &_down_ptrs_suh,
        &_down_ptrs_svh,
        &_expert_count,
        &_token_sorted,
        &_weight_sorted,
        (void*) &hidden_dim,
        (void*) &intermediate_dim,
        (void*) &num_experts,
        (void*) &num_experts_per_tok,
        (void*) &max_tokens_per_expert,
        (void*) &concurrency,
        (void*) &act_limit,
        (void*) &act_function,
        (void*) &K_gate,
        (void*) &K_up,
        (void*) &K_down,
        (void*) &locks
    };

    cudaLaunchKernel
    (
        (void*) kernel,
        grid_dim,
        block_dim,
        kernelArgs,
        smem,
        stream
    );

    cuda_check(cudaPeekAtLastError());
}
