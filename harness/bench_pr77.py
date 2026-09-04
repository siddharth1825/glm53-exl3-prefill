#!/usr/bin/env python3
"""Step 0 of the kernel plan: PR77 (E2 fat-expert kernels) image, A/B on our box.
spec 14 = spec 11 + image glm53-flash-sm121:pr77 + EXL3_FAT_KERNEL=1 at MNBT 2048 (kernel-only delta vs config 4)
spec 15 = same at MNBT 7168 (upstream default)
Runs the same bench_matrix phases as configs 4/4b and merges rows 4c/4d into bench/results.json."""
import json, sys
import bench_matrix as bm
IMAGE = "glm53-flash-sm121:pr77"
specs = bm.gw("GET", "/admin/specs")
base = [s for s in specs if s["id"] == 11][0]

def make(name, mnbt, fat="1", workspace="stock", gmu=None, grouped=False):
    spec = json.loads(json.dumps(base["spec"]))
    spec["name"] = name; spec["image"] = IMAGE
    spec["env"].update({"EXL3_FAT_KERNEL": fat, "EXL3_FAT_SORTED": fat, "EXL3_FAT_BATCHED": fat,
                        "GLM53_INDEXER_WORKSPACE": workspace, "GLM53_SPINWAIT_MS": "stock",
                        "MAX_NUM_BATCHED_TOKENS": str(mnbt)})
    if gmu: spec["env"]["GPU_MEM_UTIL"] = gmu
    if grouped:
        spec["env"].update({"EXL3_GROUPED_PREFILL": "1", "EXL3_GROUPED_SRC": "/opt/glm53/grouped/exl3_grouped_prefill.cu", "EXL3_GROUPED_BUILD": "/opt/glm53/grouped/build_grouped"})
        spec["volumes"] = spec["volumes"] + ["/home/<user>/glm-opt/fatv2:/opt/glm53/grouped", "/home/<user>/glm-opt/fatv2/patch_exl3_grouped.py:/opt/glm53/patch_exl3_grouped.py:ro"]
    have = [s for s in specs if s["name"] == name]
    body = {"name": name, "icon": "⚡", "priority": "elastic", "spec": spec}
    if have:
        bm.gw("PATCH", f"/admin/specs/{have[0]['id']}", body); return have[0]["id"]
    return bm.gw("POST", "/admin/specs", body)["id"]

s14 = make("glm-exl3-pr77-2048", 2048); s15 = make("glm-exl3-pr77-7168", 7168, workspace="rightsize", gmu="0.87"); s16 = make("glm-exl3-pr77img-legacy", 2048, fat="0"); s17 = make("glm-exl3-grouped-7168", 7168, workspace="rightsize", gmu="0.87", grouped=True)
bm.log(f"specs ready: {s14} (MNBT 2048), {s15} (MNBT 7168), image {IMAGE}")
cfgs = [
    {"label": "4c-glm-exl3-dflash-PR77-mnbt2048", "spec": s14, "port": 8888, "model": "GLM-5.3-Flash-EXL3", "notes": "PR77 E2 fat-expert kernels, EXL3_FAT_KERNEL=1, MNBT 2048 (kernel-only delta vs config 4)"},
    {"label": "4d-glm-exl3-dflash-PR77-mnbt7168", "spec": s15, "port": 8888, "model": "GLM-5.3-Flash-EXL3", "notes": "PR77 E2 fat-expert kernels, MNBT 7168 + GLM53_INDEXER_WORKSPACE=rightsize + GPU_MEM_UTIL 0.87 (upstream validated 1M boot; stock workspace at 0.84 left 10.55 GiB for KV, rightsize at 0.84 13.33 GiB, 1M needs 14.52 GiB)"},
    {"label": "4f-glm-exl3-dflash-GROUPED-mnbt7168", "spec": s17, "port": 8888, "model": "GLM-5.3-Flash-EXL3", "notes": "design 1 grouped single-launch prefill MoE (EXL3_GROUPED_PREFILL=1) on top of 4d's config"},
    {"label": "4e-glm-exl3-dflash-PR77img-legacy-mnbt2048", "spec": s16, "port": 8888, "model": "GLM-5.3-Flash-EXL3", "notes": "same pr77 image, EXL3_FAT_KERNEL=0 (legacy fat path): isolates the kernel from other image changes"},
]
only = sys.argv[1] if len(sys.argv) > 1 else None
path = f"{bm.OUT}/results.json"
for cfg in cfgs:
    if only and not cfg["label"].startswith(only): continue
    gpu = bm.GpuSampler(); gpu.start()
    res = bm.run_config(cfg, gpu); gpu.stop.set()
    allres = [x for x in json.load(open(path)) if not x["config"].startswith(cfg["label"][:2])] + [res]
    allres.sort(key=lambda x: x["config"]); json.dump(allres, open(path, "w"), indent=1)
    bm.log(f"saved {cfg['label']}")
bm.log("PR77 BENCH COMPLETE")
