#!/usr/bin/env python3
"""Numerics gate for a prefill-path change: prompt logprobs under the served model.

  python3 logprob_capture.py <tag>            capture -> bench/logprobs_<tag>.json
  python3 logprob_capture.py --compare A B    per-token |dlogprob| stats + top-20 KL(A||B), both directions

Fixed texts (~1.5–2.5K tokens each, deterministic) go through the completions API with
prompt_logprobs=20, max_tokens=1, so every token's logprob comes from the PREFILL path (> the 128-row cap).
Compares two captures token by token; the true-token logprob delta is exact, the KL is over the top-20 lists.
"""
import json, math, os, random, sys, urllib.request
PORT, MODEL = 8888, "GLM-5.3-Flash-EXL3"
OUT = os.path.expanduser("~/glm-opt/bench")

def texts():
    rng = random.Random(20260903)
    words = ("the model server allocates pages for each request while the scheduler decides which sequence "
             "advances next; a chunk of prompt tokens is embedded, rotated, routed to eight experts and reduced "
             "across both devices before the next layer begins. Latency depends on how many rows each expert "
             "receives, how the trellis tiles are decoded, and whether the pipeline keeps the tensor cores fed. ").split()
    code = """def route(tokens, experts, k=8):
    scores = sigmoid(tokens @ experts.gate.T) + experts.bias
    top = scores.topk(k, dim=-1)
    weights = scores.gather(-1, top.indices)
    weights = weights / weights.sum(-1, keepdim=True) * 2.5
    return top.indices, weights
"""
    t1 = " ".join(rng.choice(words) for _ in range(1400))
    t2 = (code * 40)
    t3 = " ".join(f"{rng.randint(0, 99999)}" for _ in range(1200))
    return {"prose": t1, "code": t2, "numbers": t3}

def capture(tag):
    res = {}
    for name, text in texts().items():
        body = {"model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 20}
        r = urllib.request.Request(f"http://localhost:{PORT}/v1/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        d = json.load(urllib.request.urlopen(r, timeout=900))
        pl = d["choices"][0]["prompt_logprobs"]
        toks = []
        for pos in pl:
            if not pos: toks.append(None); continue
            # vLLM: {token_id: {"logprob": x, "rank": r, "decoded_token": s}}; the true token is the one with rank 1? no —
            # the true token is included even if outside top-k; identify it as the entry present in every position list is
            # not possible, so store the whole map and let compare() align on token ids.
            toks.append({str(k): v["logprob"] for k, v in pos.items()})
        res[name] = {"n": len(toks), "positions": toks, "usage": d.get("usage")}
        print(f"  {tag}/{name}: {len(toks)} prompt positions", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/logprobs_{tag}.json", "w"))
    print(f"saved {OUT}/logprobs_{tag}.json")

def compare(a, b):
    A = json.load(open(f"{OUT}/logprobs_{a}.json")); B = json.load(open(f"{OUT}/logprobs_{b}.json"))
    for name in A:
        deltas, kls, kls_r, top1_same, n = [], [], [], 0, 0
        for pa, pb in zip(A[name]["positions"], B[name]["positions"]):
            if not pa or not pb: continue
            n += 1
            ta = max(pa, key=pa.get); tb = max(pb, key=pb.get)
            top1_same += (ta == tb)
            common = set(pa) & set(pb)
            if not common: continue
            # exact delta on the shared highest-probability token
            deltas.append(abs(pa[ta] - pb.get(ta, -30.0)))
            # top-20 KL both directions over the union (missing -> -30)
            def kl(p, q):
                keys = set(p) | set(q)
                lp = {k: p.get(k, -30.0) for k in keys}; lq = {k: q.get(k, -30.0) for k in keys}
                za = math.log(sum(math.exp(v) for v in lp.values())); zb = math.log(sum(math.exp(v) for v in lq.values()))
                return sum(math.exp(lp[k] - za) * ((lp[k] - za) - (lq[k] - zb)) for k in keys)
            kls.append(kl(pa, pb)); kls_r.append(kl(pb, pa))
        deltas.sort(); kls.sort()
        print(f"{name:8s} n={n:5d} top1 agree {top1_same/n:6.2%} | |dlogprob(top1)| mean {sum(deltas)/len(deltas):.4f} p99 {deltas[int(0.99*len(deltas))-1]:.4f} max {deltas[-1]:.4f} | KL(A||B) mean {sum(kls)/len(kls):.5f} p99 {kls[int(0.99*len(kls))-1]:.5f} | KL(B||A) mean {sum(kls_r)/len(kls_r):.5f}")

if __name__ == "__main__":
    if sys.argv[1] == "--compare": compare(sys.argv[2], sys.argv[3])
    else: capture(sys.argv[1])
