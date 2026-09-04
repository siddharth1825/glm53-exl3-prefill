#!/usr/bin/env python3
"""Grouped single-launch prefill MoE (design 1, grouped form) vs the overlay's reference loop and its
current fused/E2 path, on one real GLM-5.3-Flash MoE layer (all 288 experts, TP rank 0), skewed routing.

  --build-only   compile only (no GPU)
  --tokens       chunk sizes to test/time
"""
import argparse, json, os, re, statistics, sys, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/raid/models/GLM-5.3-Flash-EXL3-TR3-4bpw")
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--tp", type=int, default=2); ap.add_argument("--tp-rank", type=int, default=0)
ap.add_argument("--experts", type=int, default=0, help="0 = all")
ap.add_argument("--tokens", default="1024,2048,4096,8192")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--build-only", action="store_true")
ap.add_argument("--src", default="/work/fatv2/exl3_grouped_prefill.cu")
ap.add_argument("--build-dir", default="/work/fatv2/build_grouped")
args = ap.parse_args()

EXT_ROOT = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"
PIP_INC = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
CUDA_INC = "/usr/local/cuda/include"
SHIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shim")
os.makedirs(SHIM, exist_ok=True)
for fn in os.listdir(PIP_INC):
    src = os.path.join(PIP_INC, fn)
    if os.path.isfile(src) and not os.path.exists(os.path.join(CUDA_INC, fn)) and not os.path.lexists(os.path.join(SHIM, fn)):
        os.symlink(src, os.path.join(SHIM, fn))
for k in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"): os.environ.pop(k, None)
os.makedirs(args.build_dir, exist_ok=True)
from torch.utils.cpp_extension import load
t0 = time.time()
gp = load(name="exl3_grouped_prefill", sources=[args.src], extra_include_paths=[EXT_ROOT, SHIM],
          extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a", "-Xptxas", "-v", "--use_fast_math"],
          extra_cflags=["-O3"], build_directory=args.build_dir, verbose=True)
print(f"built/loaded exl3_grouped_prefill in {time.time()-t0:.0f}s", flush=True)
if args.build_only: sys.exit(0)

sys.path.insert(0, "/opt/glm53")
os.environ.setdefault("EXL3_FAT_KERNEL", "1")
import vllm.model_executor.layers.quantization.exl3 as ov
import exllamav3_ext as ext
dev = torch.device("cuda:0")

# ---------------------------------------------------------------- load the layer's experts, stacked like the vLLM layer
cfg = json.load(open(f"{args.model}/config.json")); cfg = cfg.get("text_config", cfg)
hidden = int(cfg["hidden_size"]); n_routed = int(cfg.get("n_routed_experts") or cfg.get("num_experts")); topk = int(cfg.get("num_experts_per_tok") or 8)
E = args.experts or n_routed
index = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
pat = re.compile(rf"^model\.(?:language_model\.)?layers\.{args.layer}\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mcg)$")
wanted = {}
for name, shard in index.items():
    m = pat.match(name)
    if m and int(m.group(1)) < E:
        wanted.setdefault(shard, []).append((name, int(m.group(1)), m.group(2), m.group(3)))
from safetensors import safe_open
raw = {}
t0 = time.time()
for shard, items in wanted.items():
    with safe_open(f"{args.model}/{shard}", framework="pt", device="cpu") as f:
        for name, e, proj, suf in items:
            t = f.get_tensor(name)
            t = ov.shard_exl3_col(t, suf, args.tp_rank, args.tp) if proj != "down_proj" else ov.shard_exl3_row(t, suf, args.tp_rank, args.tp)
            raw[(e, proj, suf)] = t.to(dev)
print(f"loaded {E} experts in {time.time()-t0:.0f}s", flush=True)
inners = []
for e in range(E):
    pack = {}
    for proj, key in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        pack[key] = ov.make_linear_exl3(raw[(e, proj, "trellis")], raw[(e, proj, "suh")], raw[(e, proj, "svh")], raw[(e, proj, "mcg")], out_dtype=torch.float16)
    inners.append(pack)
I = int(inners[0]["gate"].out_features); K = hidden
w13_trellis = torch.stack([torch.stack([raw[(e, "gate_proj", "trellis")], raw[(e, "up_proj", "trellis")]]) for e in range(E)]).contiguous()  # [E,2,tk,tn,64]
w13_suh = torch.stack([torch.stack([raw[(e, "gate_proj", "suh")], raw[(e, "up_proj", "suh")]]) for e in range(E)]).contiguous()             # [E,2,K]
w13_svh = torch.stack([torch.stack([raw[(e, "gate_proj", "svh")], raw[(e, "up_proj", "svh")]]) for e in range(E)]).contiguous()             # [E,2,I]
w2_trellis = torch.stack([raw[(e, "down_proj", "trellis")] for e in range(E)]).contiguous()                                                 # [E,tk,tn,64]
w2_suh = torch.stack([raw[(e, "down_proj", "suh")] for e in range(E)]).contiguous()                                                         # [E,I]
w2_svh = torch.stack([raw[(e, "down_proj", "svh")] for e in range(E)]).contiguous()                                                         # [E,H]
shared_suh = bool(torch.equal(w13_suh[:, 0], w13_suh[:, 1]))
print(f"stacked: w13_trellis {tuple(w13_trellis.shape)} w2_trellis {tuple(w2_trellis.shape)}; gate.suh == up.suh: {shared_suh}", flush=True)
assert shared_suh, "gate|up stacked GEMM needs a shared input sign vector (PR77 E1/E2 assume the same)"
LIMIT = float(ov.SWIGLU_LIMIT_DEFAULT)
for x in (raw,): del x


class FakeLayer(torch.nn.Module): pass
layer = FakeLayer(); layer.w13_trellis = w13_trellis; layer._exl3_hidden_size = hidden; layer._exl3_intermediate_local = I; layer._exl3_bits = 4; layer.expert_map = None
ov.build_exl3_fused_state(layer, inners)
print(f"overlay fused state ready (concurrency={layer._exl3_fused_concurrency}); fat tier configured: {ov.configured_fat_tier()}", flush=True)


def routing_skewed(T, seed, hot=12, share=0.55):
    """top-8 ids per token; `hot` experts take `share` of all slots (real prefill routing is skewed)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.empty(T, topk, dtype=torch.long)
    for t in range(T):
        pool = []
        while len(pool) < topk:
            e = int(torch.randint(0, hot, (1,), generator=g)) if torch.rand(1, generator=g).item() < share else int(torch.randint(0, E, (1,), generator=g))
            if e not in pool: pool.append(e)
        ids[t] = torch.tensor(pool)
    w = torch.rand(T, topk, generator=g); w = w / w.sum(-1, keepdim=True)
    return ids.to(dev), w.to(dev)


class Grouped:
    """The proposed prefill path: all-GPU, one launch per stage."""
    def __init__(self):
        self.max_blocks = None
    def __call__(self, x16, ids, wts, timing=False):
        T = x16.shape[0]; R = T * topk
        flat_expert = ids.reshape(-1)
        order = torch.argsort(flat_expert, stable=True)
        token_sorted = torch.arange(T, device=dev).repeat_interleave(topk)[order].contiguous()
        weight_sorted = wts.reshape(-1)[order].to(torch.float16).contiguous()
        expert_of_row = flat_expert[order].to(torch.int32).contiguous()
        counts = torch.bincount(flat_expert, minlength=E).to(torch.int32)
        expert_row0 = (torch.cumsum(counts, 0) - counts).to(torch.int32).contiguous()
        max_blocks = R // 128 + E
        block_expert = torch.empty(max_blocks, dtype=torch.int32, device=dev); block_row0 = torch.empty_like(block_expert); meta = torch.zeros(2, dtype=torch.int32, device=dev)
        gp.build_blocks(counts, block_expert, block_row0, meta)
        h13 = torch.empty(R, K, dtype=torch.float16, device=dev)
        gp.row_had(x16, h13, token_sorted, expert_of_row, w13_suh[:, 0], True)
        gu = torch.empty(R, 2 * I, dtype=torch.float16, device=dev)
        gp.grouped_gemm(h13, w13_trellis[:, 0], w13_trellis[:, 1], w13_svh[:, 0], w13_svh[:, 1], block_expert, block_row0, counts, expert_row0, token_sorted, weight_sorted, gu, 0)
        act = torch.empty(R, I, dtype=torch.float16, device=dev)
        gp.act_had(gu, act, expert_of_row, w2_suh, LIMIT)
        out = torch.zeros(T, hidden, dtype=torch.float32, device=dev)
        gp.grouped_gemm(act, w2_trellis, None, w2_svh, None, block_expert, block_row0, counts, expert_row0, token_sorted, weight_sorted, out, 1)
        return out


grouped = Grouped()

def rel(a, b): return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()

def timeit(fn):
    for _ in range(2): fn()
    ts = []
    for _ in range(args.reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)

print("\n== correctness (skewed routing) ==")
for T in (256, 2048):
    ids, wts = routing_skewed(T, 7)
    x16 = (torch.randn(T, K, device=dev) * 0.5).half().contiguous()
    ref = ov.apply_exl3_python_loop(x16, ids, wts, inners, None, LIMIT)
    got = grouped(x16, ids, wts)
    cur = ov.apply_exl3_fused_moe(x16, ids, wts, layer, inners, None, LIMIT)
    print(f"  T={T:5d}: grouped vs reference rel L2 {rel(got, ref):.2e} | current overlay path ({layer._exl3_last_fat_fallback}) vs reference {rel(cur, ref):.2e} | max |grouped-ref| {(got-ref).abs().max().item():.3e}", flush=True)

print(f"\n== per-layer time (ms, median of {args.reps}), all {E} experts, skewed routing ==")
print(f"{'tokens':>7} | {'current overlay (fused + E2 fat)':>34} | {'grouped (new)':>14} | speedup")
for T in [int(t) for t in args.tokens.split(",")]:
    ids, wts = routing_skewed(T, T)
    x16 = (torch.randn(T, K, device=dev) * 0.5).half().contiguous()
    t_cur = timeit(lambda: ov.apply_exl3_fused_moe(x16, ids, wts, layer, inners, None, LIMIT))
    t_new = timeit(lambda: grouped(x16, ids, wts))
    print(f"{T:>7} | {t_cur:>34.2f} | {t_new:>14.2f} | {t_cur/t_new:.2f}x", flush=True)
print("GROUPED TEST COMPLETE")
