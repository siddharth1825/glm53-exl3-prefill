#!/usr/bin/env python3
"""Op-level test for the EXL3 fused MoE kernel at real GLM-5.3-Flash shapes.

Runs INSIDE the MiaAI EXL3 image (needs exllamav3_ext + vllm importable), on one
GPU, with one real MoE layer's routed-expert tensors loaded straight from the
checkpoint shards — no vLLM engine, no model boot.

Reference = apply_exl3_python_loop: the overlay's per-expert LinearEXL3 path,
an independent numeric route (trellis GEMM for small M, reconstruct + cuBLAS
above LinearEXL3's threshold). The fused kernel — original (M=16 passes) and
patched (M=64 passes, selected by the host when bsz > 16; EXL3_MOE_PREFILL_M=16
forces the original) — must match it.

Checks, in order (rule 1: correctness before speed):
  1. fat-expert double-count: a routing where one expert gets > temp rows must
     still match the reference (kernel must skip it, python loop must add it).
  2. random top-8 routing at bsz 64 / 1024 / 2048 / 4096 vs reference.
  3. only then: timing, median of 5, CUDA-event, per bsz, for whichever .so is
     loaded (run once with the image's .so, once with the rebuilt one).

Usage (in container):
  python3 test_exl3_moe_op.py --model /raid/models/GLM-5.3-Flash-EXL3-TR3-4bpw --layer 3 [--tp-rank 0 --tp 2]
"""
import argparse, glob, json, os, re, statistics, sys, time

import torch

sys.path.insert(0, "/opt/glm53")  # image's overlay dir; or point at a copy via --overlay
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--layer", type=int, default=3, help="first MoE layer is 3 (layers 0-2 dense)")
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--tp-rank", type=int, default=0)
ap.add_argument("--overlay", default=None, help="path to an exl3.py to import instead of the image's")
ap.add_argument("--experts", type=int, default=0, help="limit experts loaded (0 = all 288)")
ap.add_argument("--temp-rows", type=int, default=128)
ap.add_argument("--no-timing", action="store_true")
ap.add_argument("--only-bsz", type=int, default=0, help="profiler mode: skip correctness, run --reps fused calls at this bsz (skewed routing)")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--bench-reconstruct", action="store_true", help="time exllamav3_ext.reconstruct (trellis->fp16) at real expert shapes")
args = ap.parse_args()

if args.overlay:
    import importlib.util
    spec = importlib.util.spec_from_file_location("exl3_overlay", args.overlay)
    ov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ov)
else:
    # the image installs the overlay as vllm's exl3 quantization module
    import vllm.model_executor.layers.quantization.exl3 as ov

os.environ.setdefault("EXL3_TEMP_ROWS_FUSED", str(args.temp_rows))
dev = torch.device("cuda:0")

# ------------------------------------------------------------ load one layer's experts
cfg = json.load(open(f"{args.model}/config.json"))
cfg = cfg.get("text_config", cfg)  # multimodal wrapper: the LM config is nested
hidden = int(cfg["hidden_size"])
inter = int(cfg.get("moe_intermediate_size") or cfg["intermediate_size"])
n_routed = int(cfg.get("n_routed_experts") or cfg.get("num_experts"))
topk = int(cfg.get("num_experts_per_tok") or cfg.get("moe_topk") or 8)
n_load = args.experts or n_routed
print(f"model: hidden={hidden} moe_intermediate={inter} experts={n_routed} top_k={topk}; loading {n_load} experts of layer {args.layer}, tp={args.tp} rank={args.tp_rank}")

index = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
pat = re.compile(rf"^model\.(?:language_model\.)?layers\.{args.layer}\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mcg)$")
wanted = {}
for name, shard in index.items():
    m = pat.match(name)
    if m and int(m.group(1)) < n_load:
        wanted.setdefault(shard, []).append((name, int(m.group(1)), m.group(2), m.group(3)))
if not wanted:
    # fall back: print what expert-ish names exist so the regex can be fixed
    ex = [n for n in index if f"layers.{args.layer}." in n and "expert" in n][:8]
    sys.exit(f"no expert tensors matched for layer {args.layer}; examples: {ex}")

from safetensors import safe_open
raw = {}  # (expert, proj, suffix) -> tensor (already sharded for TP)
for shard, items in wanted.items():
    with safe_open(f"{args.model}/{shard}", framework="pt", device="cpu") as f:
        for name, e, proj, suf in items:
            t = f.get_tensor(name)
            if proj in ("gate_proj", "up_proj"):
                t = ov.shard_exl3_col(t, suf, args.tp_rank, args.tp)
            else:
                t = ov.shard_exl3_row(t, suf, args.tp_rank, args.tp)
            raw[(e, proj, suf)] = t.to(dev)
print(f"loaded {len(raw)} tensors from {len(wanted)} shards")

inners = []
for e in range(n_load):
    pack = {}
    for proj, key in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        pack[key] = ov.make_linear_exl3(raw[(e, proj, "trellis")], raw[(e, proj, "suh")],
                                        raw[(e, proj, "svh")], raw[(e, proj, "mcg")], out_dtype=torch.float16)
    inners.append(pack)
inter_local = int(inners[0]["gate"].out_features)
print(f"inners built: {len(inners)} experts, intermediate_local={inter_local}")


class FakeLayer(torch.nn.Module):
    pass


layer = FakeLayer()
layer.w13_trellis = raw[(0, "gate_proj", "trellis")]        # only .device is read
layer._exl3_hidden_size = hidden
layer._exl3_intermediate_local = inter_local
layer._exl3_bits = 4
layer.expert_map = None
ov.build_exl3_fused_state(layer, inners)
print(f"fused state: concurrency={layer._exl3_fused_concurrency} temp_rows={layer._exl3_fused_temps[0].shape[1]}")

import exllamav3_ext
print("exllamav3_ext:", exllamav3_ext.__file__)
LIMIT = ov.SWIGLU_LIMIT_DEFAULT


def routing(bsz, seed, fat_expert=None, fat_rows=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(n_load, generator=g)[:topk] for _ in range(bsz)]).to(dev)
    if fat_expert is not None:
        # force one expert onto the first fat_rows tokens (slot 0)
        ids[:fat_rows, 0] = fat_expert
        # keep rows unique per token
        for r in range(fat_rows):
            dup = (ids[r, 1:] == fat_expert).nonzero()
            for d in dup.view(-1).tolist():
                ids[r, 1 + d] = (fat_expert + 1 + d) % n_load
    w = torch.rand(bsz, topk, generator=g).to(dev)
    w = w / w.sum(-1, keepdim=True)
    return ids, w


def routing_skewed(bsz, seed, hot=12, share=0.55):
    """Zipf-like: `share` of all slots land on `hot` experts. Mimics real MoE routing
    where MiaAI measured max_rows ~ the whole chunk on the hottest expert."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.empty(bsz, topk, dtype=torch.long)
    for r in range(bsz):
        pool = []
        while len(pool) < topk:
            if torch.rand(1, generator=g).item() < share:
                e = int(torch.randint(0, hot, (1,), generator=g))
            else:
                e = int(torch.randint(hot, n_load, (1,), generator=g))
            if e not in pool: pool.append(e)
        ids[r] = torch.tensor(pool)
    w = torch.rand(bsz, topk, generator=g); w = w / w.sum(-1, keepdim=True)
    return ids.to(dev), w.to(dev)


def run_ref(x, ids, w):
    return ov.apply_exl3_python_loop(x, ids, w, inners, None, LIMIT)


def run_fused(x, ids, w):
    return ov.apply_exl3_fused_moe(x, ids, w, layer, inners, None, LIMIT)


def compare(tag, a, b):
    d = (a - b).abs()
    denom = b.abs().max().clamp(min=1e-6)
    print(f"  {tag:<34} max_abs={d.max().item():.4e} mean_abs={d.mean().item():.4e} rel_to_max={d.max().item()/denom.item():.3e}")
    return d.max().item() / denom.item()


torch.manual_seed(0)
if args.bench_reconstruct:
    import exllamav3_ext as ext
    # one expert's three matrices, TP-sharded shapes: gate/up K=4096 N=1024, down K=1024 N=4096
    packs = [inners[e] for e in range(8)]
    def recon_all(pack):
        for key in ("gate", "up", "down"):
            lin = pack[key]
            w = torch.empty((lin.in_features, lin.out_features), dtype=torch.half, device=dev)
            ext.reconstruct(w, lin.trellis, lin.K, lin.mcg, lin.mul1)
    recon_all(packs[0]); torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record()
        for pk in packs: recon_all(pk)
        en.record(); torch.cuda.synchronize(); ts.append(st.elapsed_time(en))
    med = statistics.median(ts)
    weights = 8 * (4096*1024*2 + 1024*4096)
    out_bytes = weights * 2; in_bytes = weights // 2
    print(f"reconstruct: 8 experts x 3 matrices = {weights/1e6:.0f}M weights in {med:.2f} ms")
    print(f"  -> {weights/med/1e6:,.0f} M weights/ms ; fp16 write {out_bytes/med/1e6:,.0f} MB/ms ; packed read {in_bytes/med/1e6:,.0f} MB/ms")
    per_layer = 288/8 * med
    print(f"  -> all 288 experts of one layer (this node's TP half): {per_layer:.0f} ms/layer  (DRAM floor ~ {(288*(4096*1024*2+1024*4096)*2.5)/273e6:.0f} ms at 273 GB/s)")
    sys.exit(0)
if args.only_bsz:
    bsz = args.only_bsz
    x = (torch.randn(bsz, hidden, device=dev) * 0.5).half().float()
    ids, w = routing_skewed(bsz, 4242)
    run_fused(x, ids, w); torch.cuda.synchronize()   # warm (JIT/attrs)
    for _ in range(args.reps):
        run_fused(x, ids, w)
    torch.cuda.synchronize()
    print(f"profile mode: {args.reps} fused calls at bsz={bsz} done")
    sys.exit(0)
fails = 0
TOL = 5e-3  # baseline run measured ~9e-4 rel between the two routes; 5x margin

# 1. fat-expert double-count check
bsz = 512
x = (torch.randn(bsz, hidden, device=dev) * 0.5).half().float()
ids, w = routing(bsz, 1, fat_expert=7, fat_rows=400)   # expert 7 gets 400 rows > temp_rows 128
ref = run_ref(x, ids, w); torch.cuda.synchronize()
out = run_fused(x, ids, w); torch.cuda.synchronize()
r = compare(f"fat check bsz={bsz} expert7={400}rows", out, ref)
fails += r > TOL

# 2. random routing at prefill sizes
for bsz in (16, 64, 1024, 2048, 4096):
    x = (torch.randn(bsz, hidden, device=dev) * 0.5).half().float()
    ids, w = routing(bsz, 100 + bsz)
    ref = run_ref(x, ids, w); torch.cuda.synchronize()
    out = run_fused(x, ids, w); torch.cuda.synchronize()
    counts = torch.bincount(ids.view(-1), minlength=n_load)
    r = compare(f"random bsz={bsz} max_rows={counts.max().item()}", out, ref)
    fails += r > TOL

x = (torch.randn(2048, hidden, device=dev) * 0.5).half().float()
ids, w = routing_skewed(2048, 777)
ref = run_ref(x, ids, w); torch.cuda.synchronize()
out = run_fused(x, ids, w); torch.cuda.synchronize()
counts = torch.bincount(ids.view(-1), minlength=n_load)
r = compare(f"SKEWED bsz=2048 max_rows={counts.max().item()}", out, ref)
fails += r > TOL

print("CORRECTNESS:", "PASS" if not fails else f"FAIL ({fails})")
if fails or args.no_timing:
    sys.exit(1 if fails else 0)

# 3. timing — only after outputs match
print(f"\nTIMING (median of 5, CUDA events) — EXL3_MOE_PREFILL_M={os.environ.get('EXL3_MOE_PREFILL_M','auto')}")
for bsz, kind in ((64,"uniform"), (256,"uniform"), (1024,"uniform"), (2048,"uniform"), (4096,"uniform"), (1024,"skewed"), (2048,"skewed"), (4096,"skewed")):
    x = (torch.randn(bsz, hidden, device=dev) * 0.5).half().float()
    ids, w = routing(bsz, 500 + bsz) if kind == "uniform" else routing_skewed(bsz, 900 + bsz)
    for _ in range(2):
        run_fused(x, ids, w)
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); run_fused(x, ids, w); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    med = statistics.median(ts)
    mr = torch.bincount(ids.view(-1), minlength=n_load).max().item()
    print(f"  bsz={bsz:>5} {kind:<8} max_rows={mr:>5}  {med:8.2f} ms/layer   {bsz*topk/med:9.0f} token-slots/ms   ({min(ts):.2f}-{max(ts):.2f})")
