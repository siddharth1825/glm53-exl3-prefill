#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_m16_s3f2() { return exl3_moe_kernel<4, 256, 16, 1, 3, 2>; }
