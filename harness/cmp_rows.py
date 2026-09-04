import json, statistics as st, sys
R={r["config"]:r for r in json.load(open("bench/results.json"))}
def row(c):
    p=R[c]["phases"]; g=lambda ph,k: [r[k] for r in p[ph]["runs"]]
    return dict(cold32K=st.median(g("cold_prefill_32k","prefill_tok_s")), ttft32K=st.median(g("cold_prefill_32k","ttft_s")),
                cold8K=st.median(g("cold_prefill_8k","prefill_tok_s")), warm8K=st.median(g("warm_prefill_8k","prefill_tok_s")),
                prose=st.median(g("decode_prose","decode_tok_s")), code=st.median(g("decode_code","decode_tok_s")))
pref = tuple(sys.argv[1:]) or ("4-","4b","4c","4d")
for c in sorted(R):
    if c.startswith(pref):
        r=row(c); print(f"{c:38s} cold32K {r['cold32K']:6.0f} tok/s (TTFT {r['ttft32K']:5.1f}s)  cold8K {r['cold8K']:5.0f}  warm8K {r['warm8K']:5.0f}  decode prose {r['prose']:4.1f} code {r['code']:4.1f}")
