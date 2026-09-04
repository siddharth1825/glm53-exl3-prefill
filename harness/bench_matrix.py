#!/usr/bin/env python3
"""GB10 serving benchmark matrix — DeepSeek vs GLM lanes, prefill + decode + GPU.

Runs on spark-head, measures against the ENGINE'S LOCAL PORT (no gateway, no
Cloudflare, no tailnet in the number). Config switching goes through the
gateway API because it owns the multi-node orchestration.

Per config:
  0. launch, wait until a real generation succeeds
  1. JIT burn-in: 2 x 400-token generations, DISCARDED
     (first-request kernel/TileLang JIT is 30-50 s on these lanes and has
      already caused two published retractions in this community)
  2. cold prefill: 3 x ~32K unique prompts + 3 x ~8K unique prompts
     unique salt at position 0 => guaranteed prefix-cache miss
  3. warm prefill: repeat the last 8K prompt verbatim => cache hit
  4. decode: 3 x prose, 3 x code, 400 max_tokens, temp 0, thinking off
  5. /metrics delta (speculation accept rates where exposed)
GPU sampled at 1 Hz on both nodes for the whole run; phases are timestamped
and joined afterwards.

Numbers reported as medians. prefill tok/s = prompt_tokens / TTFT (streamed).
decode tok/s = completion_tokens / (total - TTFT).
"""
import json, os, random, socket, statistics, subprocess, sys, threading, time, urllib.error, urllib.request

# The Sparks resolve <GATEWAY_HOST> to a Cloudflare IPv6 address whose path
# is blackholed — connects hang in SYN-SENT. Pin every outbound socket to IPv4.
_gai = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *a, **kw: _gai(host, port, socket.AF_INET, *a, **kw)

GW = "https://<GATEWAY_HOST>"
TOKEN = open(os.path.expanduser("~/.gwtoken")).read().strip()
HDR = {"Authorization": f"Bearer {TOKEN}", "content-type": "application/json", "User-Agent": "curl/8.0"}
OUT = os.path.expanduser("~/glm-opt/bench")
WORKER = "192.168.99.2"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- gateway API
def gw(method, path, body=None, timeout=60, tries=4):
    data = json.dumps(body).encode() if body is not None else None
    last = None
    for i in range(tries):
        try:
            r = urllib.request.Request(GW + path, data=data, headers=HDR, method=method)
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError:
            raise                      # a real answer from the gateway: let the caller decide
        except Exception as e:         # transport flake: retry
            last = e
            log(f"    gateway {method} {path} transport error ({type(e).__name__}) — retry {i+1}/{tries}")
            time.sleep(5)
    raise RuntimeError(f"gateway unreachable after {tries} tries: {last}")


def stop_everything():
    for d in gw("GET", "/admin/deployments"):
        if d["status"] in ("running", "launching"):
            log(f"    stopping deployment {d['id']} ({d['name']})")
            try:
                gw("POST", f"/admin/deployments/{d['id']}/stop")
            except Exception as e:
                log(f"    stop failed: {e}")
    time.sleep(30)  # let heartbeats report the freed memory


def patch_spec_env(spec_id, env_updates, extra_args=None):
    row = [s for s in gw("GET", "/admin/specs") if s["id"] == spec_id][0]
    spec = row["spec"]
    spec.setdefault("env", {}).update(env_updates)
    if extra_args is not None:
        spec["extra_args"] = extra_args
    gw("PATCH", f"/admin/specs/{spec_id}",
       {"name": row["name"], "icon": row.get("icon"), "priority": row.get("priority"), "spec": spec})


def launch(spec_id, tries=12):
    for i in range(tries):
        try:
            return gw("POST", f"/admin/specs/{spec_id}/launch", {})
        except urllib.error.HTTPError as e:
            log(f"    launch attempt {i+1} refused ({e.code}) — waiting for placement")
            time.sleep(20)
    raise RuntimeError("launch never accepted")


# ---------------------------------------------------------------- measurement
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(f"{OUT}/run.log", "a") as f:
        f.write(line + "\n")


WORDS = ("the quick brown fox jumps over lazy dog while ancient rivers carve deep valleys through "
         "granite mountains and merchants trade silk spices along dusty roads between distant cities "
         "where scholars debate philosophy under olive trees as farmers harvest golden wheat beneath "
         "an endless summer sky and children chase shadows across cobblestone squares near fountains "
         "carved by forgotten artisans whose names dissolved into the patient accumulation of years "
         "though their work endures in stone and memory alike sustaining travellers who pause to rest "
         "before continuing toward harbours where ships wait with folded sails for favourable winds").split()


def unique_prompt(n_words, seed):
    """Cache-hostile prose: unique salt first, shuffled real words after."""
    rng = random.Random(seed)
    body = [rng.choice(WORDS) for _ in range(n_words)]
    salt = f"DOCUMENT-{seed}-{rng.getrandbits(48):x}"
    return (f"{salt}\n\n" + " ".join(body) +
            "\n\nIn one short sentence, what is the general subject of the text above?")


def chat(port, model, messages, max_tokens, stream=True, timeout=900):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    first = None
    usage = None
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if not stream:
            d = json.load(resp)
            return {"ttft": time.time() - t0, "total": time.time() - t0,
                    "usage": d.get("usage", {}), "text": d["choices"][0]["message"].get("content", "")}
        for line in resp:
            if not line.startswith(b"data: ") or b"[DONE]" in line:
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch:
                delta = ch[0].get("delta", {}) or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    n_chunks += 1
                    if first is None:
                        first = time.time()
    t1 = time.time()
    return {"ttft": (first or t1) - t0, "total": t1 - t0, "usage": usage or {}, "chunks": n_chunks}


def metrics(port):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=15) as r:
            return r.read().decode()
    except Exception:
        return ""


def spec_counters(raw):
    out = {}
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        low = line.lower()
        if any(k in low for k in ("spec_", "accept", "draft")):
            try:
                name, val = line.rsplit(" ", 1)
                out[name.split("{")[0]] = float(val)
            except ValueError:
                pass
    return out


# ---------------------------------------------------------------- gpu sampler
SAMPLER = (
    "nvidia-smi --query-gpu=clocks.current.sm,temperature.gpu,power.draw,"
    "utilization.gpu,pstate --format=csv,noheader,nounits -l 1"
)


class GpuSampler:
    """1 Hz on both nodes; rows are (epoch, node, clock, temp, power, util, pstate)."""

    def __init__(self):
        self.rows = []
        self.stop = threading.Event()
        self.threads = []

    def _run(self, node, cmd):
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            for line in p.stdout:
                if self.stop.is_set():
                    break
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 5:
                    try:
                        self.rows.append((time.time(), node, float(parts[0]), float(parts[1]),
                                          float(parts[2]), float(parts[3]), parts[4]))
                    except ValueError:
                        pass
        finally:
            p.kill()

    def start(self):
        self.threads = [
            threading.Thread(target=self._run, args=("head", SAMPLER), daemon=True),
            threading.Thread(target=self._run, args=("worker", f"ssh -o BatchMode=yes {WORKER} '{SAMPLER}'"), daemon=True),
        ]
        for t in self.threads:
            t.start()

    def summarize(self, t0, t1):
        out = {}
        for node in ("head", "worker"):
            sel = [r for r in self.rows if node == r[1] and t0 <= r[0] <= t1]
            if not sel:
                out[node] = None
                continue
            out[node] = {
                "samples": len(sel),
                "clock_mhz": {"median": statistics.median(r[2] for r in sel), "max": max(r[2] for r in sel)},
                "temp_c": {"median": statistics.median(r[3] for r in sel), "max": max(r[3] for r in sel)},
                "power_w": {"median": statistics.median(r[4] for r in sel), "max": max(r[4] for r in sel)},
                "util_pct": {"median": statistics.median(r[5] for r in sel), "max": max(r[5] for r in sel)},
                "pstates": sorted(set(r[6] for r in sel)),
            }
        return out


def capture_provenance():
    """Image, argv and serving env of the container that is actually running.

    Captured live because the gateway deletes a stopped deployment's row when a
    same-named one launches — so reading it back afterwards loses configs whose
    successor reused the name (which is every config in a matrix like this).
    """
    try:
        names = subprocess.check_output(
            "docker ps --filter name=gw- --format '{{.Names}}'", shell=True, text=True).split()
        if not names:
            return {"error": "no gw- container running"}
        name = names[0]
        def d(fmt):
            return subprocess.check_output(
                f"docker inspect {name} --format '{fmt}'", shell=True, text=True).strip()
        argv = subprocess.check_output(
            f"docker exec {name} sh -c \"tr '\\0' ' ' < /proc/1/cmdline\"",
            shell=True, text=True, timeout=60).strip()
        env = subprocess.check_output(
            f"docker inspect {name} --format '{{{{range .Config.Env}}}}{{{{println .}}}}{{{{end}}}}'",
            shell=True, text=True).splitlines()
        return {
            "container": name,
            "image": d("{{.Config.Image}}"),
            "image_id": d("{{.Image}}"),
            "argv": argv,
            "env": sorted(e for e in env if e and not e.startswith(("PATH=", "LD_", "LS_COLORS"))),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def host_mem():
    out = {}
    for node, cmd in (("head", "free -m"), ("worker", f"ssh -o BatchMode=yes {WORKER} free -m")):
        try:
            line = [l for l in subprocess.check_output(cmd, shell=True, text=True).splitlines()
                    if l.startswith("Mem:")][0].split()
            out[node] = {"total_mb": int(line[1]), "used_mb": int(line[2]), "available_mb": int(line[6])}
        except Exception:
            out[node] = None
    return out


# ---------------------------------------------------------------- one config
def wait_serving(port, model, budget_s=1500):
    t0 = time.time()
    while time.time() - t0 < budget_s:
        try:
            r = chat(port, model, [{"role": "user", "content": "hi"}], 4, stream=False, timeout=60)
            if r["usage"]:
                log(f"    serving after {time.time()-t0:.0f}s")
                return True
        except Exception:
            pass
        time.sleep(15)
    return False


def run_config(cfg, gpu):
    res = {"config": cfg["label"], "spec": cfg["spec"], "port": cfg["port"],
           "model": cfg["model"], "notes": cfg.get("notes", ""), "phases": {}}
    log(f"=== {cfg['label']} ===")
    stop_everything()
    if cfg.get("env") or cfg.get("extra_args") is not None:
        patch_spec_env(cfg["spec"], cfg.get("env", {}), cfg.get("extra_args"))
    t_launch = time.time()
    d = launch(cfg["spec"])
    res["deployment"] = d.get("id")
    log(f"    deployment {d.get('id')} launching")
    if not wait_serving(cfg["port"], cfg["model"]):
        res["error"] = "never became servable"
        return res
    res["boot_seconds"] = round(time.time() - t_launch)
    res["provenance"] = capture_provenance()

    # 1. JIT burn-in, discarded
    log("    burn-in (discarded)")
    for _ in range(2):
        try:
            chat(cfg["port"], cfg["model"],
                 [{"role": "user", "content": "Write a short paragraph about rivers."}], 400)
        except Exception as e:
            log(f"    burn-in error: {e}")

    m0 = spec_counters(metrics(cfg["port"]))

    def phase(name, fn, n):
        t0 = time.time()
        rows = []
        for i in range(n):
            try:
                rows.append(fn(i))
            except Exception as e:
                log(f"    {name}[{i}] failed: {e}")
        t1 = time.time()
        res["phases"][name] = {"runs": rows, "gpu": gpu.summarize(t0, t1)}
        return rows

    # 2. cold prefill — unique prompts, cache-hostile
    seed_base = int(time.time())

    def cold(n_words):
        def one(i):
            p = unique_prompt(n_words, seed_base + i * 7919 + n_words)
            r = chat(cfg["port"], cfg["model"], [{"role": "user", "content": p}], 48)
            pt = r["usage"].get("prompt_tokens", 0)
            return {"prompt_tokens": pt, "ttft_s": round(r["ttft"], 2),
                    "prefill_tok_s": round(pt / r["ttft"], 1) if r["ttft"] > 0 else None}
        return one

    log("    cold prefill 32K x3")
    phase("cold_prefill_32k", cold(24000), 3)
    log("    cold prefill 8K x3")
    phase("cold_prefill_8k", cold(6000), 3)

    # 3. warm prefill — repeat the last 8K prompt verbatim (cache hit)
    log("    warm prefill (cache hit)")
    warm_prompt = unique_prompt(6000, seed_base + 2 * 7919 + 6000)

    def warm(i):
        r = chat(cfg["port"], cfg["model"], [{"role": "user", "content": warm_prompt}], 48)
        pt = r["usage"].get("prompt_tokens", 0)
        return {"prompt_tokens": pt, "ttft_s": round(r["ttft"], 2),
                "prefill_tok_s": round(pt / r["ttft"], 1) if r["ttft"] > 0 else None}

    phase("warm_prefill_8k", warm, 2)

    # 4. decode
    def decode(prompt):
        def one(i):
            r = chat(cfg["port"], cfg["model"], [{"role": "user", "content": prompt}], 400)
            ct = r["usage"].get("completion_tokens", 0)
            gen = r["total"] - r["ttft"]
            return {"completion_tokens": ct, "ttft_s": round(r["ttft"], 2),
                    "total_s": round(r["total"], 2),
                    "decode_tok_s": round(ct / gen, 1) if gen > 0 else None}
        return one

    log("    decode prose x3")
    phase("decode_prose", decode(
        "Explain in plain prose how a hash map works, including collisions and resizing."), 3)
    log("    decode code x3")
    phase("decode_code", decode(
        "Write a Python function that merges two sorted lists into one sorted list, "
        "with type hints and a docstring."), 3)

    m1 = spec_counters(metrics(cfg["port"]))
    res["metrics_delta"] = {k: round(m1[k] - m0.get(k, 0), 2)
                            for k in m1 if m1[k] != m0.get(k, 0)}
    res["host_mem"] = host_mem()
    return res


CONFIGS = [
    {"label": "1-deepseek-v4-flash-fp8", "spec": 6, "port": 8000, "model": "deepseek-v4-0731",
     "notes": "incumbent; different model/tokenizer — compare wall-clock too"},
    {"label": "2-glm-exl3-base", "spec": 11, "port": 8888, "model": "GLM-5.3-Flash-EXL3",
     "env": {"SPEC_METHOD": "none"}, "notes": "no speculation"},
    {"label": "3-glm-exl3-mtp", "spec": 11, "port": 8888, "model": "GLM-5.3-Flash-EXL3",
     "env": {"SPEC_METHOD": "mtp", "MTP_TOKENS": "2"}},
    {"label": "4-glm-exl3-dflash", "spec": 11, "port": 8888, "model": "GLM-5.3-Flash-EXL3",
     "env": {"SPEC_METHOD": "dflash", "DFLASH_MODEL_DIR": "/raid/models/GLM-5.3-Flash-DFlash2",
             "DFLASH_TOKENS": "7", "DFLASH_DRAFT_TP": "2"}},
    {"label": "5-glm-nvfp4-sglang-base", "spec": 9, "port": 8100, "model": "GLM-5.3-Flash",
     "extra_args": ['--attention-backend', 'dsa', '--dsa-prefill-backend', 'tilelang', '--dsa-decode-backend', 'tilelang', '--moe-runner-backend', 'flashinfer_cutlass', '--disable-shared-experts-fusion', '--max-running-requests', '2'],
     "notes": "SGLang lane, dealignai NVFP4 (spec 9 production args pinned)"},
    {"label": "6-glm-nvfp4-sglang-eagle-mtp", "spec": 9, "port": 8100, "model": "GLM-5.3-Flash",
     "extra_args": ["--attention-backend", "dsa", "--dsa-prefill-backend", "tilelang",
                    "--dsa-decode-backend", "tilelang", "--moe-runner-backend", "flashinfer_cutlass",
                    "--disable-shared-experts-fusion", "--max-running-requests", "2",
                    "--speculative-algorithm", "EAGLE", "--speculative-eagle-topk", "1",
                    "--speculative-num-steps", "5", "--speculative-num-draft-tokens", "6"],
     "notes": "STRETCH: native MTP on SGLang, never measured publicly"},
]

if __name__ == "__main__":
    only = sys.argv[1:] or None
    gpu = GpuSampler()
    gpu.start()
    results = []
    for cfg in CONFIGS:
        if only and not any(o in cfg["label"] for o in only):
            continue
        try:
            r = run_config(cfg, gpu)
        except Exception as e:
            r = {"config": cfg["label"], "error": f"{type(e).__name__}: {e}"}
            log(f"    CONFIG FAILED: {e}")
        results.append(r)
        with open(f"{OUT}/results.json", "w") as f:
            json.dump(results, f, indent=1)
        log(f"    wrote {OUT}/results.json ({len(results)} configs)")
    gpu.stop.set()
    log("MATRIX COMPLETE")
