#!/usr/bin/env python3
"""Design-1 kernel test: exl3_fat_gemm_v2 vs PR77's exl3_fat_gemm vs the exact reference
(reconstruct -> fp32 matmul -> H128*svh), at real GLM-5.3-Flash expert shapes.

Runs inside glm53-flash-sm121:pr77. JIT-builds the v2 module against the image's exllamav3 ext headers.
  --build-only      compile and exit (no GPU needed)
  --rows            per-expert row counts to test/time (fat experts: 129..~2000 rows in a chunk)
  --variants        v2 variants: 0=K64/3 stages, 1=K64/2, 2=K32/4, 3=K32/3
"""
import argparse, json, os, re, statistics, sys, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/raid/models/GLM-5.3-Flash-EXL3-TR3-4bpw")
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--tp", type=int, default=2); ap.add_argument("--tp-rank", type=int, default=0)
ap.add_argument("--experts", type=int, default=4)
ap.add_argument("--rows", default="128,256,512,1024,2048")
ap.add_argument("--variants", default="0,1,2,3")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--build-only", action="store_true")
ap.add_argument("--src", default="/work/fatv2/exl3_fat_gemm_v2.cu")
ap.add_argument("--build-dir", default="/work/fatv2/build")
args = ap.parse_args()

EXT_ROOT = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"
# ATen needs cusparse.h/cublas*.h, which only the pip cu13 wheel ships — but that wheel's crt/host_runtime.h
# is older than nvcc 13.0.88's launch stub (1-arg vs 2-arg __cudaLaunch). So expose ONLY the wheel's
# top-level headers that /usr/local/cuda/include lacks, through a shim directory, and never its crt/.
PIP_INC = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
CUDA_INC = "/usr/local/cuda/include"
SHIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shim")
os.makedirs(SHIM, exist_ok=True)
for fn in os.listdir(PIP_INC):
    src = os.path.join(PIP_INC, fn)
    if os.path.isfile(src) and not os.path.exists(os.path.join(CUDA_INC, fn)) and not os.path.lexists(os.path.join(SHIM, fn)):
        os.symlink(src, os.path.join(SHIM, fn))
os.environ.pop("CPATH", None); os.environ.pop("CPLUS_INCLUDE_PATH", None); os.environ.pop("C_INCLUDE_PATH", None)
os.makedirs(args.build_dir, exist_ok=True)
from torch.utils.cpp_extension import load
t0 = time.time()
ext2 = load(name="exl3_fat_v2", sources=[args.src], extra_include_paths=[EXT_ROOT, SHIM],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a", "-Xptxas", "-v", "--use_fast_math"],
            extra_cflags=["-O3"], build_directory=args.build_dir, verbose=True)
print(f"built/loaded exl3_fat_v2 in {time.time()-t0:.0f}s; smem per variant:", [int(ext2.smem_bytes(v)) for v in range(4)], flush=True)
if args.build_only:
    sys.exit(0)

sys.path.insert(0, "/opt/glm53")
import vllm.model_executor.layers.quantization.exl3 as ov
import exllamav3_ext as ext
dev = torch.device("cuda:0")

# ---------------------------------------------------------------- load experts (same loader as the op harness)
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
inners = []
for e in range(args.experts):
    pack = {}
    for proj, key in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        pack[key] = ov.make_linear_exl3(raw[(e, proj, "trellis")], raw[(e, proj, "suh")], raw[(e, proj, "svh")], raw[(e, proj, "mcg")], out_dtype=torch.float16)
    inners.append(pack)
g0 = inners[0]["gate"]; I = int(g0.out_features); H = int(g0.in_features)
print(f"experts loaded: {args.experts}; gate/up K={H} N_local={I} (fused 2I={2*I}); down K={I} N={H}", flush=True)


def fused13(e):
    gate, up = inners[e]["gate"], inners[e]["up"]
    packed13 = torch.cat([gate.trellis, up.trellis], dim=1).contiguous()
    svh13 = torch.cat([gate.svh, up.svh]).contiguous()
    return packed13, svh13, gate


def reference(h, packed, svh, lin):
    """Exact path: reconstruct -> fp32 matmul -> H128 * svh (what E2's fallback does, in fp32)."""
    K, N = packed.shape[0] * 16, packed.shape[1] * 16
    w = torch.empty(K, N, dtype=torch.float16, device=dev)
    ext.reconstruct(w, packed, lin.K, lin.mcg, lin.mul1)
    y = h.float() @ w.float()
    y = y.contiguous()
    ext.had_r_128(y, y, None, svh, 1.0)
    return y


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def timeit(fn):
    for _ in range(3): fn()
    ts = []
    for _ in range(args.reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)


variants = [int(v) for v in args.variants.split(",")]
rows_list = [int(r) for r in args.rows.split(",")]
torch.manual_seed(0)

# ---------------------------------------------------------------- correctness (gate|up and down, direct + scatter)
print("\n== correctness (expert 0, M=300 rows: exercises a partial last M block) ==")
packed13, svh13, gate = fused13(0)
M = 300
h13 = (torch.randn(M, H, device=dev) * 0.5).half().contiguous()
ref = reference(h13, packed13, svh13, gate)
old = torch.empty(M, 2 * I, dtype=torch.float32, device=dev)
ext.exl3_fat_gemm(h13, packed13, old, svh13, gate.K, gate.mcg, gate.mul1)
print(f"  E2 fat_gemm vs reference: rel L2 {rel(old, ref):.2e}")
for v in variants:
    new = torch.empty_like(old)
    ext2.fat_gemm_v2(h13, packed13, new, svh13, gate.K, gate.mcg, gate.mul1, v)
    print(f"  v2 variant {v} vs reference: rel L2 {rel(new, ref):.2e} | vs E2: {rel(new, old):.2e} | max abs diff vs E2 {(new-old).abs().max().item():.3e}")
down = inners[0]["down"]
h2 = (torch.randn(M, I, device=dev) * 0.5).half().contiguous()
refd = reference(h2, down.trellis, down.svh, down)
tok = torch.randperm(4096, device=dev)[:M].contiguous()
rw = (torch.rand(M, device=dev) * 0.5 + 0.25).half().contiguous()
old_s = torch.zeros(4096, H, dtype=torch.float32, device=dev)
ext.exl3_fat_gemm_scatter(h2, down.trellis, old_s, down.svh, tok, rw, down.K, down.mcg, down.mul1)
exp_s = torch.zeros_like(old_s); exp_s.index_add_(0, tok, refd * rw.float()[:, None])
print(f"  E2 scatter vs reference scatter: rel L2 {rel(old_s, exp_s):.2e}")
for v in variants:
    new_s = torch.zeros_like(old_s)
    ext2.fat_gemm_v2_scatter(h2, down.trellis, new_s, down.svh, tok, rw, down.K, down.mcg, down.mul1, v)
    print(f"  v2 variant {v} scatter vs reference: rel L2 {rel(new_s, exp_s):.2e} | vs E2: {rel(new_s, old_s):.2e}")

# ---------------------------------------------------------------- timing
print(f"\n== timing per expert call, median of {args.reps} (ms) and TFLOPS; gate|up: M x {H} x {2*I}; down: M x {I} x {H} ==")
hdr = f"{'rows':>6} | {'E2 13':>8} {'E2 dn':>8} | " + " | ".join(f"v{v} 13 {'':>1}v{v} dn" for v in variants)
print(hdr)
for M in rows_list:
    h13 = (torch.randn(M, H, device=dev) * 0.5).half().contiguous()
    h2 = (torch.randn(M, I, device=dev) * 0.5).half().contiguous()
    out13 = torch.empty(M, 2 * I, dtype=torch.float32, device=dev)
    outd = torch.empty(M, H, dtype=torch.float32, device=dev)
    fl13 = 2 * M * H * 2 * I; fld = 2 * M * I * H
    e2_13 = timeit(lambda: ext.exl3_fat_gemm(h13, packed13, out13, svh13, gate.K, gate.mcg, gate.mul1))
    e2_d = timeit(lambda: ext.exl3_fat_gemm(h2, down.trellis, outd, down.svh, down.K, down.mcg, down.mul1))
    line = f"{M:>6} | {e2_13:>8.3f} {e2_d:>8.3f} | "
    cells = []
    for v in variants:
        v13 = timeit(lambda: ext2.fat_gemm_v2(h13, packed13, out13, svh13, gate.K, gate.mcg, gate.mul1, v))
        vd = timeit(lambda: ext2.fat_gemm_v2(h2, down.trellis, outd, down.svh, down.K, down.mcg, down.mul1, v))
        cells.append(f"{v13:>7.3f} {vd:>7.3f}")
    print(line + " | ".join(cells) + f"   [E2 TFLOPS 13/dn: {fl13/e2_13/1e9:.0f}/{fld/e2_d/1e9:.0f}]", flush=True)
# per-layer projection: 288 experts, avg rows = tokens*8/288 -> use the closest measured rows
print("\n(per-layer estimate = 288 x (13 + dn) at the row count closest to tokens*8/288; compare with fused trellis 24 ms @2K, 110 ms @8K)")
print("TEST COMPLETE")
