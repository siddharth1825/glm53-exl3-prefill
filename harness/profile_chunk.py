#!/usr/bin/env python3
"""Profile the served GLM EXL3 engine on a cold 8K prompt with vLLM's built-in torch profiler.
spec 13 = spec 11 + VLLM_TORCH_PROFILER_DIR=/raid/prof (both ranks write traces to /raid/prof)."""
import json, os, random, time, urllib.request
import bench_matrix as bm
PORT, MODEL = 8888, "GLM-5.3-Flash-EXL3"
specs = bm.gw("GET", "/admin/specs")
base = [s for s in specs if s["id"] == 11][0]
spec = json.loads(json.dumps(base["spec"])); spec["name"] = "glm-exl3-prof"
spec["env"]["VLLM_TORCH_PROFILER_DIR"] = "/raid/prof"
have = [s for s in specs if s["name"] == "glm-exl3-prof"]
if have:
    sid = have[0]["id"]; bm.gw("PATCH", f"/admin/specs/{sid}", {"name": "glm-exl3-prof", "icon": "🔬", "priority": "elastic", "spec": spec})
else:
    sid = bm.gw("POST", "/admin/specs", {"name": "glm-exl3-prof", "icon": "🔬", "priority": "elastic", "spec": spec})["id"]
bm.log(f"spec {sid} glm-exl3-prof ready")
bm.stop_everything()
d = bm.launch(sid); bm.log(f"deployment {d.get('id')} launching")
if not bm.wait_serving(PORT, MODEL): bm.log("never became servable"); raise SystemExit(1)

def cold_prompt(n_tokens, salt):
    rng = random.Random(salt)
    words = ["alpha","bridge","cobalt","delta","ember","falcon","granite","harbor","iris","jasper","kelp","lumen","marble","nectar","orbit","pewter","quartz","ridge","saffron","timber","umber","velvet","willow","xenon","yarrow","zephyr"]
    body = " ".join(rng.choice(words) + str(rng.randint(0, 9999)) for _ in range(int(n_tokens * 0.55)))
    return f"[{salt}] Read this list and reply with the single word OK.\n{body}\nReply: "

def chat(prompt, max_tokens=8):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, "stream": False}
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(r, timeout=900)); return time.time() - t, d.get("usage", {})

el, us = chat(cold_prompt(8000, "warm-jit")); bm.log(f"warm-up (discarded): {el:.1f}s usage={us}")
el, us = chat(cold_prompt(8000, "warm-jit-2")); bm.log(f"warm-up 2 (discarded): {el:.1f}s usage={us}")
urllib.request.urlopen(urllib.request.Request(f"http://localhost:{PORT}/start_profile", data=b"", method="POST"), timeout=60).read()
bm.log("profiler started")
el, us = chat(cold_prompt(8000, "profiled-8k")); bm.log(f"profiled cold 8K: {el:.1f}s usage={us}")
urllib.request.urlopen(urllib.request.Request(f"http://localhost:{PORT}/stop_profile", data=b"", method="POST"), timeout=600).read()
bm.log("profiler stopped; waiting for trace flush"); time.sleep(20)
os.system("ls -la /raid/prof/ | tail -5")
bm.log("PROFILE COMPLETE")
