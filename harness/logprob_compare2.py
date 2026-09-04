#!/usr/bin/env python3
"""Exact per-token gate: the TRUE prompt token's logprob under capture A vs B (no top-k membership artifacts).
Tokenizes the same fixed texts with the model tokenizer to know the true token at every position.
usage: python3 logprob_compare2.py A B [C ...]   (all pairs vs A)"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logprob_capture import texts
OUT = os.environ.get("LOGPROB_OUT", os.path.expanduser("~/glm-opt/bench"))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/raid/models/GLM-5.3-Flash-EXL3-TR3-4bpw")
caps = {t: json.load(open(f"{OUT}/logprobs_{t}.json")) for t in sys.argv[1:]}
A = sys.argv[1]
for name, text in texts().items():
    ids = tok(text, add_special_tokens=True)["input_ids"]
    for B in sys.argv[2:]:
        pa, pb = caps[A][name]["positions"], caps[B][name]["positions"]
        n = min(len(pa), len(pb), len(ids))
        d, la, lb, miss = [], [], [], 0
        for i in range(1, n):
            if not pa[i] or not pb[i]: continue
            k = str(ids[i])
            if k not in pa[i] or k not in pb[i]: miss += 1; continue
            la.append(pa[i][k]); lb.append(pb[i][k]); d.append(abs(pa[i][k] - pb[i][k]))
        d.sort()
        ppl_a = math.exp(-sum(la)/len(la)); ppl_b = math.exp(-sum(lb)/len(lb))
        print(f"{name:8s} {A} vs {B}: n={len(d):5d} (missing {miss}) | true-token |dlogprob| mean {sum(d)/len(d):.4f} median {d[len(d)//2]:.4f} p99 {d[int(0.99*len(d))-1]:.4f} max {d[-1]:.4f} | PPL {ppl_a:.3f} vs {ppl_b:.3f} ({(ppl_b/ppl_a-1)*100:+.2f}%)")
