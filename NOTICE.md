# Notices

- The trellis decode (`dq_dispatch`), the PTX helpers, the Hadamard routines, and the epilogue structure come from
  **exllamav3** (turboderp-org, MIT), pinned at commit c5d9c657 as vendored by the MiaAI image. The kernels here
  include its headers at build time and do not redistribute them.
- `exl3_fat_gemm_v2.cu` keeps the contract, checks, and epilogue of **PR77 `exl3_fat_gemm.cu`** from
  MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (E2 fat-expert prefill) and replaces its mainloop.
  `serving/gw-start.sh` mirrors that repository's `start.sh` runtime patch order.
- The EXL3 checkpoint used for all measurements is `GLM-5.3-Flash-EXL3-TR3-4bpw` (ShapleyMcg); see the gateway
  repository's `docs/THIRD_PARTY_NOTICES.md` for its schedule.
- The DFlash2 drafter (`incoai/GLM-5.3-Flash-DFlash2`) is CC BY-NC-ND.
- Research reports under `docs/research/` cite their sources inline.
