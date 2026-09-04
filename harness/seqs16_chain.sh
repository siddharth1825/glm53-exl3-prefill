#!/usr/bin/env bash
cd ~/glm-opt
log() { echo "[$(date +%H:%M:%S)] $*"; }
python3 - <<'PY'
import bench_matrix as bm, json, time
specs = bm.gw("GET", "/admin/specs")
base = [s for s in specs if s["id"] == 17][0]
spec = json.loads(json.dumps(base["spec"])); spec["name"] = "glm-exl3-grouped-7168-seqs16"
spec["env"].update({"MAX_NUM_SEQS": "16", "EXTRA_ARGS": '--long-prefill-token-threshold 2048 --compilation-config {"max_cudagraph_capture_size":128}'})
have = [s for s in specs if s["name"] == spec["name"]]
body = {"name": spec["name"], "icon": "⚡", "priority": "elastic", "spec": spec}
if have:
    sid = have[0]["id"]; bm.gw("PATCH", f"/admin/specs/{sid}", body)
else:
    sid = bm.gw("POST", "/admin/specs", body)["id"]
print("spec", sid, spec["name"])
bm.stop_everything(); time.sleep(45)
d = bm.launch(sid); print("deployment", d.get("id"), "launching")
ok = bm.wait_serving(8888, "GLM-5.3-Flash-EXL3")
print("SERVING" if ok else "NEVER SERVED")
PY
C=$(docker ps -q --filter name=seqs16 | head -1)
docker logs $C 2>&1 | grep -iE "max-num-seqs|long-prefill|max_cudagraph|GPU KV cache size|Maximum concurrency|Available KV cache memory|CUDA graph memory|cudagraph_capture_sizes|ERROR" | sed "s/.*\] //" | cut -c1-200 | head -10
log "burst test on seqs16"
python3 burst_test.py seqs16 4 30000 2>&1 | tee bench/burst_seqs16.log
log "8-stream burst"
python3 burst_test.py seqs16x8 8 30000 2>&1 | tee bench/burst_seqs16x8.log
log "SEQS16 CHAIN COMPLETE"
