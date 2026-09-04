import json
R={r["config"]:r for r in json.load(open("bench/results.json"))}
for c in sorted(R):
    if not c.startswith(("4-","4b","4c","4d")): continue
    r=R[c]; p=r["phases"]
    code=[(x["decode_tok_s"], x["total_s"], x["ttft_s"]) for x in p["decode_code"]["runs"]]
    prose=[x["decode_tok_s"] for x in p["decode_prose"]["runs"]]
    md=r.get("metrics_delta",{})
    acc=md.get("vllm:spec_decode_num_accepted_tokens_total"); dr=md.get("vllm:spec_decode_num_draft_tokens_total"); nd=md.get("vllm:spec_decode_num_drafts_total")
    print(f"{c:36s} code runs (tok/s,total_s,ttft): {code}  prose: {prose}")
    print(f"{'':36s} spec: accepted {acc} / drafted {dr} over {nd} drafts -> {acc/nd if nd else 0:.2f} accepted per step, {acc/dr if dr else 0:.2%} acceptance")
    gpu=p["decode_code"].get("gpu",{}); print(f"{'':36s} gpu head clock {gpu.get('head',{}).get('clock_mhz',{}).get('median')} MHz power {gpu.get('head',{}).get('power_w',{}).get('median')} W util {gpu.get('head',{}).get('util_pct',{}).get('median')}")
