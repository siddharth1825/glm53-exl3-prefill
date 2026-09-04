#pragma once

#include "../exl3_moe_common.cuh"

typedef void (*fp_exl3_moe_kernel) (EXL3_MOE_KERNEL_ARGS);

#define EXL3_MOE_DECLARE_GETTERS(K) \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256(); \

EXL3_MOE_DECLARE_GETTERS(0);
EXL3_MOE_DECLARE_GETTERS(1);
EXL3_MOE_DECLARE_GETTERS(2);
EXL3_MOE_DECLARE_GETTERS(3);
EXL3_MOE_DECLARE_GETTERS(4);
EXL3_MOE_DECLARE_GETTERS(5);
EXL3_MOE_DECLARE_GETTERS(6);
EXL3_MOE_DECLARE_GETTERS(7);
EXL3_MOE_DECLARE_GETTERS(8);

// Prefill (M-tiled) instances: M in {32,64} x N in {128,256}, runtime-K and 4-bit.
fp_exl3_moe_kernel exl3_moe_kernel_k4_n128_m32();
fp_exl3_moe_kernel exl3_moe_kernel_k4_n128_m64();
fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m32();
fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m64();
fp_exl3_moe_kernel exl3_moe_kernel_k0_n128_m32();
fp_exl3_moe_kernel exl3_moe_kernel_k0_n128_m64();
fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m16_occ2();
fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m32_occ2();
fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m16_s3f2();
fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m16_s4f3();
fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m16_s6f3();
fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m16_s6f2();
fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m16_s8f2();

#undef EXL3_MOE_DECLARE_GETTERS

extern fp_exl3_moe_kernel exl3_moe_kernel_instances[];
