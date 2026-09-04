#!/usr/bin/env python3
"""Validate the tuned EXL3 MoE kernel through the real gates (rule 1).

1. Create spec 12 = spec 11 (glm-exl3-dflash) + the rebuilt exllamav3_ext bind-mounted
   over the image's copy + env EXL3_MOE_SMEM_FIT=1 EXL3_MOE_STAGES=s8f2.
2. Temp-0 output diff: boot spec 11 (stock kernel), capture 12 fixed prompts (thinking ON,
   256 max_tokens); boot spec 12 (tuned), capture the same; compare token-for-token.
3. Full bench_matrix phases on spec 12 (cold 32K/8K prefill, warm, decode) — comparable to
   yesterday's config 4 numbers.
"""
import json, os, sys, time
import bench_matrix as bm

SO_IN_CONTAINER = "/usr/local/lib/python3.12/dist-packages/exllamav3_ext.cpython-312-aarch64-linux-gnu.so"
SO_HOST = "/home/<user>/glm-opt/exl3build/exllamav3_ext_s8f2.so"
PORT, MODEL = 8888, "GLM-5.3-Flash-EXL3"
PROMPTS = [
    "What is 19*21? Reply with just the number.",
    "Write a Python function that merges two sorted lists into one sorted list, with type hints.",
    "Explain in plain prose how a hash map handles collisions.",
    "List three differences between TCP and UDP as bullet points.",
    "Translate to French: 'The library closes at nine on weekdays.'",
    "Given the JSON {\"a\": 3, \"b\": [1,2,3]}, what is the sum of all numbers? Show your reasoning briefly.",
    "Write a haiku about a GPU kernel.",
    "What is the derivative of x^3 * sin(x)?",
    "Summarize the plot of Hamlet in two sentences.",
    "Write a bash one-liner that counts lines in all .py files under the current directory.",
    "A train leaves at 3:15pm and the trip takes 2h50m. When does it arrive?",
    "Name the capital of Australia and one fact about it.",
]

# ---- 1. spec 12
specs = bm.gw("GET", "/admin/specs")
base = [s for s in specs if s["id"] == 11][0]
have = [s for s in specs if s["name"] == "glm-exl3-kern"]
spec = json.loads(json.dumps(base["spec"]))
spec["name"] = "glm-exl3-kern"
spec["volumes"] = [v for v in spec["volumes"] if SO_IN_CONTAINER not in v] + [f"{SO_HOST}:{SO_IN_CONTAINER}:ro"]
spec["env"].update({"EXL3_MOE_SMEM_FIT": "1", "EXL3_MOE_STAGES": "s8f2", "EXL3_MOE_PREFILL_M": "16"})
if have:
    sid = have[0]["id"]
    bm.gw("PATCH", f"/admin/specs/{sid}", {"name": "glm-exl3-kern", "icon": "⚡", "priority": "elastic", "spec": spec})
else:
    sid = bm.gw("POST", "/admin/specs", {"name": "glm-exl3-kern", "icon": "⚡", "priority": "elastic", "spec": spec})["id"]
bm.log(f"spec {sid} glm-exl3-kern ready (tuned kernel bind-mounted)")


def capture(spec_id, tag):
    bm.stop_everything()
    d = bm.launch(spec_id); bm.log(f"[{tag}] deployment {d.get('id')} launching")
    if not bm.wait_serving(PORT, MODEL):
        bm.log(f"[{tag}] never became servable"); return None
    # warm the JIT so the diff is not contaminated by first-request effects
    bm.chat(PORT, MODEL, [{"role": "user", "content": "warm up"}], 64, stream=False)
    out = {}
    for p in PROMPTS:
        body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 256, "temperature": 0, "stream": False}
        import urllib.request
        r = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        d = json.load(urllib.request.urlopen(r, timeout=600))
        m = d["choices"][0]["message"]
        out[p] = {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "", "usage": d.get("usage")}
    json.dump(out, open(f"{bm.OUT}/temp0_{tag}.json", "w"), indent=1)
    bm.log(f"[{tag}] captured {len(out)} prompts")
    return out


stock = capture(11, "stock")
tuned = capture(sid, "tuned")
if stock and tuned:
    same = sum(1 for p in PROMPTS if stock[p]["content"] == tuned[p]["content"] and stock[p]["reasoning"] == tuned[p]["reasoning"])
    same_content = sum(1 for p in PROMPTS if stock[p]["content"] == tuned[p]["content"])
    bm.log(f"TEMP0 DIFF: identical (content+reasoning) {same}/{len(PROMPTS)}; identical content {same_content}/{len(PROMPTS)}")
    for p in PROMPTS:
        if stock[p]["content"] != tuned[p]["content"]:
            a, b = stock[p]["content"], tuned[p]["content"]
            i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
            bm.log(f"  diverges at char {i}: {p[:40]!r} | stock: {a[max(0,i-20):i+30]!r} | tuned: {b[max(0,i-20):i+30]!r}")

# ---- 3. bench on the tuned spec (same phases as the matrix)
cfg = {"label": "4b-glm-exl3-dflash-TUNED-kernel", "spec": sid, "port": PORT, "model": MODEL,
       "notes": "smem-fit + s8f2 (FRAG_STAGES 2, SH_STAGES 8) on the DFlash config"}
gpu = bm.GpuSampler(); gpu.start()
res = bm.run_config(cfg, gpu); gpu.stop.set()
path = f"{bm.OUT}/results.json"
allres = [x for x in json.load(open(path)) if not x["config"].startswith("4b-")] + [res]
allres.sort(key=lambda x: x["config"]); json.dump(allres, open(path, "w"), indent=1)
bm.log("VALIDATE COMPLETE")
