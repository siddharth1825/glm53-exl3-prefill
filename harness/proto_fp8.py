#!/usr/bin/env python3
"""Step 1 prototype for the dequant-once fp8 prefill path (GLM-5.3-Flash EXL3, real shapes).

  trellis --reconstruct--> W' --H,suh,H,svh--> W (fp32, real domain)  [exllamav3 get_weight_tensor convention]
  W --128x128 block quant--> fp8 e4m3 + fp32 scales
  vLLM Triton fused_experts (fp8 w8a8, block [128,128])  vs  overlay apply_exl3_python_loop (reference)

Gates: (1) my dequant matches LinearEXL3.forward (x @ W == forward(x)), (2) fp16-dequant MoE matches the
reference loop at fp16-noise level, (3) fp8 path error vs reference is reported (expected ~1e-2 rel L2, i.e.
fp8-checkpoint class), (4) grouped calls (expert_map) sum to the single call. Then timing.
Runs inside the MiaAI EXL3 image on one GPU. Token counts are scaled by experts/288 so per-expert row
density matches a real chunk; the per-layer extrapolation multiplies back.
"""
import argparse, json, math, re, statistics, sys
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--tp", type=int, default=2); ap.add_argument("--tp-rank", type=int, default=0)
ap.add_argument("--experts", type=int, default=96)
ap.add_argument("--tokens", default="1024,2048,4096,8192")
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()
sys.path.insert(0, "/opt/glm53")
import vllm.model_executor.layers.quantization.exl3 as ov
import exllamav3_ext as ext
from vllm.model_executor.layers.fused_moe import fused_moe as fm
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
dev = torch.device("cuda:0")
N_ROUTED = 288

cfg = json.load(open(f"{args.model}/config.json")); cfg = cfg.get("text_config", cfg)
hidden = int(cfg["hidden_size"]); topk = int(cfg.get("num_experts_per_tok") or 8)
index = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
pat = re.compile(rf"^model\.(?:language_model\.)?layers\.{args.layer}\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mcg)$")
wanted = {}
for name, shard in index.items():
    m = pat.match(name)
    if m and int(m.group(1)) < args.experts:
        wanted.setdefault(shard, []).append((name, int(m.group(1)), m.group(2), m.group(3)))
from safetensors import safe_open
raw = {}
for shard, items in wanted.items():
    with safe_open(f"{args.model}/{shard}", framework="pt", device="cpu") as f:
        for name, e, proj, suf in items:
            t = f.get_tensor(name)
            t = ov.shard_exl3_col(t, suf, args.tp_rank, args.tp) if proj != "down_proj" else ov.shard_exl3_row(t, suf, args.tp_rank, args.tp)
            raw[(e, proj, suf)] = t.to(dev)
E = args.experts
inners = []
for e in range(E):
    pack = {}
    for proj, key in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        pack[key] = ov.make_linear_exl3(raw[(e, proj, "trellis")], raw[(e, proj, "suh")], raw[(e, proj, "svh")], raw[(e, proj, "mcg")], out_dtype=torch.float16)
    inners.append(pack)
I = int(inners[0]["gate"].out_features)
LIMIT = float(ov.SWIGLU_LIMIT_DEFAULT)
print(f"loaded {E} experts of layer {args.layer}: hidden={hidden} intermediate_local={I} topk={topk} swiglu_limit={LIMIT}", flush=True)

# ---------------------------------------------------------------- full dequant (exllamav3 get_weight_tensor convention)
h = torch.ones(1, 1, device=dev)
while h.shape[0] < 128:
    h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
H128 = h / math.sqrt(128)

def full_weight(lin):
    """[K_in, N_out] fp32 real-domain weight: diag(suh) . H . W' . H . diag(svh)."""
    K, N = int(lin.in_features), int(lin.out_features)
    w = torch.empty(K, N, dtype=torch.float16, device=dev)
    ext.reconstruct(w, lin.trellis, lin.K, lin.mcg, lin.mul1)
    w = w.float()
    w = (H128 @ w.view(-1, 128, N)).view(K, N)
    w = w * lin.suh.float()[:, None]
    w = (w.view(K, -1, 128) @ H128).view(K, N)
    w = w * lin.svh.float()[None, :]
    return w

# gate 1: dequant convention vs LinearEXL3.forward
torch.manual_seed(0)
x = torch.randn(64, hidden, device=dev, dtype=torch.float16)
lin = inners[0]["gate"]
ref = lin.forward(x, {}, out_dtype=torch.float32)
got = x.float() @ full_weight(lin)
err = (got - ref).norm() / ref.norm()
print(f"gate1 dequant-convention vs LinearEXL3.forward: rel L2 {err:.2e}  ({'PASS' if err < 5e-3 else 'FAIL'})", flush=True)
lin = inners[0]["down"]
xd = torch.randn(64, I, device=dev, dtype=torch.float16)
err = ((xd.float() @ full_weight(lin)) - lin.forward(xd, {}, out_dtype=torch.float32)).norm() / lin.forward(xd, {}, out_dtype=torch.float32).norm()
print(f"gate1 (down_proj): rel L2 {err:.2e}  ({'PASS' if err < 5e-3 else 'FAIL'})", flush=True)

# ---------------------------------------------------------------- build fp8 block-scaled expert tensors
def block_fp8(w):
    R, C = w.shape
    b = w.view(R // 128, 128, C // 128, 128).permute(0, 2, 1, 3)
    amax = b.abs().amax(dim=(-1, -2)).clamp(min=1e-12)
    scale = amax / 448.0
    q = (b / scale[..., None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.permute(0, 2, 1, 3).reshape(R, C).contiguous(), scale.contiguous()

w1 = torch.empty(E, 2 * I, hidden, dtype=torch.float8_e4m3fn, device=dev)
w2 = torch.empty(E, hidden, I, dtype=torch.float8_e4m3fn, device=dev)
w1s = torch.empty(E, 2 * I // 128, hidden // 128, dtype=torch.float32, device=dev)
w2s = torch.empty(E, hidden // 128, I // 128, dtype=torch.float32, device=dev)
Wfull = []  # fp16 full weights for the fp16-dequant reference (kept for a few experts only to save memory)
t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
t0.record()
for e in range(E):
    g = full_weight(inners[e]["gate"]); u = full_weight(inners[e]["up"]); d = full_weight(inners[e]["down"])
    w1[e, :I], w1s[e, :I // 128] = block_fp8(g.T.contiguous())
    w1[e, I:], w1s[e, I // 128:] = block_fp8(u.T.contiguous())
    w2[e], w2s[e] = block_fp8(d.T.contiguous())
    Wfull.append((g.half(), u.half(), d.half()))
t1.record(); torch.cuda.synchronize()
print(f"dequant+blockquant (torch, unoptimised) {E} experts: {t0.elapsed_time(t1):.0f} ms -> {t0.elapsed_time(t1)*N_ROUTED/E:.0f} ms per full layer", flush=True)
qcfg = fp8_w8a8_moe_quant_config(w1_scale=w1s, w2_scale=w2s, block_shape=[128, 128])

def routing(T, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(T)]).to(dev)
    w = torch.rand(T, topk, generator=g).to(dev); w = w / w.sum(-1, keepdim=True)
    return ids, w

def ref_fp16(x16, ids, wts):
    out = torch.zeros(x16.shape[0], hidden, dtype=torch.float32, device=dev)
    for e in range(E):
        tok, kpos = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0: continue
        hcur = x16.index_select(0, tok)
        g, u, d = Wfull[e]
        gate = (hcur @ g).float(); up = (hcur @ u).float()
        act = torch.nn.functional.silu(gate.clamp(max=LIMIT)) * up.clamp(min=-LIMIT, max=LIMIT)
        down = (act.half() @ d).float()
        out.index_add_(0, tok, down * wts[tok, kpos].unsqueeze(-1).float())
    return out

def fp8_path(xb, ids, wts, expert_map=None, w1_=None, w2_=None, qc=None):
    return fm.fused_experts(xb, w1_ if w1_ is not None else w1, w2_ if w2_ is not None else w2, wts.float(), ids.to(torch.int32),
                            activation=MoEActivation.SILU, global_num_experts=E, expert_map=expert_map, quant_config=qc or qcfg)

# ---------------------------------------------------------------- gates 2-4 at a modest size
T = 512
ids, wts = routing(T, 1)
x16 = torch.randn(T, hidden, device=dev, dtype=torch.float16) * 0.5
ref = ov.apply_exl3_python_loop(x16, ids.long(), wts, inners, None, LIMIT)
r16 = ref_fp16(x16, ids, wts)
o8 = fp8_path(x16.to(torch.bfloat16), ids, wts).float()
e16 = ((r16 - ref).norm() / ref.norm()).item(); e8 = ((o8 - ref).norm() / ref.norm()).item()
print(f"gate2 fp16-dequant MoE vs reference loop: rel L2 {e16:.2e}  ({'PASS' if e16 < 5e-3 else 'FAIL'})")
print(f"gate3 fp8 block-scaled fused_experts vs reference loop: rel L2 {e8:.2e}  (fp8-checkpoint class is ~1e-2)")
# per-token worst case
pt = ((o8 - ref).norm(dim=1) / ref.norm(dim=1).clamp(min=1e-6))
print(f"       per-token rel err: median {pt.median():.2e}  p99 {pt.quantile(0.99):.2e}  max {pt.max():.2e}", flush=True)
# gate 4: grouped calls with expert_map
half = E // 2
out_g = torch.zeros_like(o8)
for lo in (0, half):
    emap = torch.full((E,), -1, dtype=torch.int32, device=dev); emap[lo:lo + half] = torch.arange(half, device=dev, dtype=torch.int32)
    qc = fp8_w8a8_moe_quant_config(w1_scale=w1s[lo:lo + half], w2_scale=w2s[lo:lo + half], block_shape=[128, 128])
    out_g += fp8_path(x16.to(torch.bfloat16), ids, wts, expert_map=emap, w1_=w1[lo:lo + half], w2_=w2[lo:lo + half], qc=qc).float()
eg = ((out_g - o8).norm() / o8.norm()).item()
print(f"gate4 two expert groups (expert_map) vs single call: rel L2 {eg:.2e}  ({'PASS' if eg < 1e-3 else 'FAIL'})", flush=True)

# ---------------------------------------------------------------- timing at real per-expert density
print(f"\ntiming (median of {args.reps}, CUDA events); tokens scaled x{E}/{N_ROUTED} for density, per-layer = x{N_ROUTED}/{E}")
for Treal in [int(t) for t in args.tokens.split(",")]:
    T = max(64, Treal * E // N_ROUTED)
    ids, wts = routing(T, Treal)
    xb = (torch.randn(T, hidden, device=dev, dtype=torch.float16) * 0.5).to(torch.bfloat16)
    for _ in range(2): fp8_path(xb, ids, wts)
    ts = []
    for _ in range(args.reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fp8_path(xb, ids, wts); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    m = statistics.median(ts)
    flops = 2 * T * topk * 3 * hidden * I
    print(f"  chunk {Treal:5d} tok (run {T:5d} tok x {E} experts): fp8 fused_experts {m:7.2f} ms -> per layer ~{m*N_ROUTED/E:6.1f} ms  ({flops/m/1e9:.0f} TFLOPS eff)", flush=True)
print("PROTO COMPLETE")
