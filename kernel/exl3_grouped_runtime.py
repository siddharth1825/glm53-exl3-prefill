"""Serving integration for the grouped single-launch EXL3 prefill MoE (design 1, grouped form).

Installed into vllm's exl3 overlay module by patch_exl3_grouped.py when EXL3_GROUPED_PREFILL=1.
Prefill calls (tokens > the fused kernel's row cap) go through one launch per stage over all experts;
decode calls (tokens <= cap) keep the original graph-safe fused path untouched.

Env:
  EXL3_GROUPED_PREFILL=1                 enable
  EXL3_GROUPED_SRC=/opt/glm53/grouped/exl3_grouped_prefill.cu
  EXL3_GROUPED_BUILD=/opt/glm53/grouped/build_grouped   (JIT cache; compiled once per image, ~30 s)
"""
import logging, os, time
import torch

logger = logging.getLogger("vllm.exl3.grouped")
_ext = None
_logged = False


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    src = os.environ.get("EXL3_GROUPED_SRC", "/opt/glm53/grouped/exl3_grouped_prefill.cu")
    build = os.environ.get("EXL3_GROUPED_BUILD", "/opt/glm53/grouped/build_grouped")
    ext_root = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"
    pip_inc = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
    cuda_inc = "/usr/local/cuda/include"
    shim = os.path.join(os.path.dirname(src), "shim")
    os.makedirs(shim, exist_ok=True)
    for fn in os.listdir(pip_inc):
        s = os.path.join(pip_inc, fn)
        if os.path.isfile(s) and not os.path.exists(os.path.join(cuda_inc, fn)) and not os.path.lexists(os.path.join(shim, fn)):
            try:
                os.symlink(s, os.path.join(shim, fn))
            except FileExistsError:
                pass
    for k in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
        os.environ.pop(k, None)
    os.makedirs(build, exist_ok=True)
    from torch.utils.cpp_extension import load
    t0 = time.time()
    _ext = load(name="exl3_grouped_prefill", sources=[src], extra_include_paths=[ext_root, shim],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a", "--use_fast_math"],
                extra_cflags=["-O3"], build_directory=build, verbose=False)
    logger.info("EXL3 grouped prefill extension loaded in %.0fs from %s", time.time() - t0, build)
    return _ext


def grouped_prefill(x2d: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, layer, limit: float) -> torch.Tensor:
    """x2d [T, K] (any float dtype), ids [T, topk] long, weights [T, topk] -> out [T, K] fp32."""
    gp = _load_ext()
    dev = x2d.device
    T, K = x2d.shape
    topk = ids.shape[1]
    R = T * topk
    E = int(layer.w13_trellis.shape[0])
    I = int(layer._exl3_intermediate_local)
    x16 = x2d.to(torch.float16).contiguous()
    flat_expert = ids.reshape(-1)
    order = torch.argsort(flat_expert, stable=True)
    token_sorted = torch.arange(T, device=dev).repeat_interleave(topk)[order].contiguous()
    weight_sorted = weights.reshape(-1)[order].to(torch.float16).contiguous()
    expert_of_row = flat_expert[order].to(torch.int32).contiguous()
    counts = torch.zeros(E, dtype=torch.int32, device=dev)
    counts.scatter_add_(0, flat_expert, torch.ones_like(flat_expert, dtype=torch.int32))
    expert_row0 = (torch.cumsum(counts, 0) - counts).to(torch.int32).contiguous()
    max_blocks = R // 128 + E
    block_expert = torch.empty(max_blocks, dtype=torch.int32, device=dev)
    block_row0 = torch.empty_like(block_expert)
    meta = torch.zeros(2, dtype=torch.int32, device=dev)
    gp.build_blocks(counts, block_expert, block_row0, meta)
    h13 = torch.empty(R, K, dtype=torch.float16, device=dev)
    gp.row_had(x16, h13, token_sorted, expert_of_row, layer.w13_suh[:, 0], True)
    gu = torch.empty(R, 2 * I, dtype=torch.float16, device=dev)
    gp.grouped_gemm(h13, layer.w13_trellis[:, 0], layer.w13_trellis[:, 1], layer.w13_svh[:, 0], layer.w13_svh[:, 1],
                    block_expert, block_row0, counts, expert_row0, token_sorted, weight_sorted, gu, 0)
    act = torch.empty(R, I, dtype=torch.float16, device=dev)
    gp.act_had(gu, act, expert_of_row, layer.w2_suh, float(limit))
    out = torch.zeros(T, K, dtype=torch.float32, device=dev)
    gp.grouped_gemm(act, layer.w2_trellis, None, layer.w2_svh, None,
                    block_expert, block_row0, counts, expert_row0, token_sorted, weight_sorted, out, 1)
    return out


def install(ov):
    """Wrap ov.apply_exl3_experts: prefill-sized calls use the grouped path; everything else is untouched."""
    global _logged
    original = ov.apply_exl3_experts

    def apply_exl3_experts_grouped(x, topk_ids, topk_weights, layer, *, limit=ov.SWIGLU_LIMIT_DEFAULT, fused=None):
        global _logged
        tokens = x.shape[-2]
        temps = getattr(layer, "_exl3_fused_temps", None)
        cap = int(temps[0].shape[1]) if temps else 0
        expert_map = getattr(layer, "expert_map", None)
        if tokens <= cap or expert_map is not None or not hasattr(layer, "w13_trellis"):
            return original(x, topk_ids, topk_weights, layer, limit=limit, fused=fused)
        x2d = x.reshape(tokens, x.shape[-1])
        ids = topk_ids.reshape(tokens, -1).to(torch.long)
        weights = topk_weights.reshape(tokens, -1)
        out = grouped_prefill(x2d, ids, weights, layer, float(limit))
        layer._exl3_last_apply = "grouped"
        if not _logged:
            logger.info("EXL3 grouped prefill engaged (tokens=%d > cap=%d, experts=%d)", tokens, cap, int(layer.w13_trellis.shape[0]))
            _logged = True
        return out.to(dtype=x.dtype)

    ov.apply_exl3_experts = apply_exl3_experts_grouped
    # the MoE method's apply() looks the function up at call time from the module globals, so the wrap sticks
    _load_ext()  # compile/load at boot rather than on the first prefill
    logger.info("EXL3 grouped prefill installed (EXL3_GROUPED_PREFILL=1)")
