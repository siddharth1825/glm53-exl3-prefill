#!/usr/bin/env python3
"""Concurrency burst test against the served GLM: N simultaneous cold ~30K-token prompts (unique salts),
streaming, measuring per-request TTFT and completion; then one single cold prompt as the baseline.
usage: python3 burst_test.py <tag> [n=4] [tokens=30000]"""
import json, random, sys, threading, time, urllib.request
PORT, MODEL = 8888, "GLM-5.3-Flash-EXL3"
tag = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4; TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 30000

def cold_prompt(n_tokens, salt):
    rng = random.Random(salt)
    words = ["alpha","bridge","cobalt","delta","ember","falcon","granite","harbor","iris","jasper","kelp","lumen","marble","nectar","orbit","pewter","quartz","ridge","saffron","timber","umber","velvet","willow","xenon","yarrow","zephyr"]
    body = " ".join(rng.choice(words) + str(rng.randint(0, 9999)) for _ in range(int(n_tokens * 0.55)))
    return f"[{salt}] Read this list and reply with the single word OK.\n{body}\nReply: "

def one(i, salt, out):
    body = {"model": MODEL, "messages": [{"role": "user", "content": cold_prompt(TOK, salt)}], "max_tokens": 16, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.time(); ttft = None; toks = 0; usage = None
    with urllib.request.urlopen(r, timeout=1800) as resp:
        for line in resp:
            if not line.startswith(b"data:"): continue
            payload = line[5:].strip()
            if payload == b"[DONE]": break
            d = json.loads(payload)
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            delta = (ch[0].get("delta") or {}) if ch else {}
            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                if ttft is None: ttft = time.time() - t0   # first generated token, thinking or answer
                toks += 1
    out[i] = {"ttft": ttft if ttft is not None else time.time() - t0, "total": time.time() - t0, "prompt_tokens": (usage or {}).get("prompt_tokens")}

single = {}
one(0, f"{tag}-single", single)
print(f"[{tag}] single cold ~{TOK}: prompt {single[0]['prompt_tokens']} tok  TTFT {single[0]['ttft']:.1f}s", flush=True)
res = {}
threads = [threading.Thread(target=one, args=(i, f"{tag}-burst-{i}", res)) for i in range(N)]
t0 = time.time()
for th in threads: th.start()
for th in threads: th.join()
wall = time.time() - t0
print(f"[{tag}] burst of {N} x ~{TOK} tokens: wall {wall:.1f}s")
for i in sorted(res):
    r = res[i]; print(f"  req {i}: prompt {r['prompt_tokens']} tok  TTFT {r['ttft']:.1f}s  total {r['total']:.1f}s")
ttfts = sorted(r["ttft"] for r in res.values())
print(f"  TTFT first {ttfts[0]:.1f}s  median {ttfts[len(ttfts)//2]:.1f}s  last {ttfts[-1]:.1f}s")
