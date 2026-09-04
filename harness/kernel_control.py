#!/usr/bin/env python3
"""Control for the temp-0 diff: same 12 prompts on the STOCK kernel a second time (raw message saved),
so we know the stock kernel's own run-to-run reproducibility before blaming the tuned one."""
import json, urllib.request, sys
import bench_matrix as bm
from kernel_validate_prompts import PROMPTS
PORT, MODEL = 8888, "GLM-5.3-Flash-EXL3"
tag = sys.argv[1] if len(sys.argv) > 1 else "stock2"; spec_id = int(sys.argv[2]) if len(sys.argv) > 2 else 11
bm.stop_everything()
d = bm.launch(spec_id); bm.log(f"[{tag}] deployment {d.get('id')} launching (spec {spec_id})")
if not bm.wait_serving(PORT, MODEL): bm.log(f"[{tag}] never became servable"); sys.exit(1)
bm.chat(PORT, MODEL, [{"role": "user", "content": "warm up"}], 64, stream=False)
out = {}
for p in PROMPTS:
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 256, "temperature": 0, "stream": False}
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    dd = json.load(urllib.request.urlopen(r, timeout=600))
    out[p] = {"message": dd["choices"][0]["message"], "usage": dd.get("usage"), "finish_reason": dd["choices"][0].get("finish_reason")}
json.dump(out, open(f"{bm.OUT}/temp0_{tag}.json", "w"), indent=1)
bm.log(f"[{tag}] captured {len(out)} prompts; message keys: {sorted(out[PROMPTS[0]]['message'].keys())}")
S = json.load(open(f"{bm.OUT}/temp0_stock.json"))
same = sum(1 for p in PROMPTS if S[p]["content"] == (out[p]["message"].get("content") or ""))
bm.log(f"CONTROL: stock-vs-{tag} identical content {same}/{len(PROMPTS)}")
for p in PROMPTS:
    a = S[p]["content"]; b = out[p]["message"].get("content") or ""
    if a != b:
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        bm.log(f"  differs at char {i}: {p[:40]!r} | stock: {a[max(0,i-20):i+30]!r} | {tag}: {b[max(0,i-20):i+30]!r}")
bm.log("CONTROL COMPLETE")
