#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k0_n128_m32() { return exl3_moe_kernel<0, 128, 32>; }
