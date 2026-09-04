#!/usr/bin/env bash
cd ~/glm-opt
log() { echo "[$(date +%H:%M:%S)] $*"; }
python3 -u - <<'PY'
import bench_matrix as bm, time
specs = bm.gw("GET", "/admin/specs"); sid = [s for s in specs if s["name"] == "glm-exl3-grouped-7168-seqs16-nothr"][0]["id"]
bm.stop_everything(); time.sleep(45)
d = bm.launch(sid); print("deployment", d.get("id"), "launching spec", sid)
print("SERVING" if bm.wait_serving(8888, "GLM-5.3-Flash-EXL3") else "NEVER SERVED")
PY
C=$(docker ps -q --filter name=nothr | head -1)
docker logs $C 2>&1 | grep -iE "GPU KV cache size|Available KV cache memory|ERROR|ValueError" | sed "s/.*\] //" | cut -c1-200 | tail -3
log "4-stream burst on seqs16 without threshold"
python3 -u burst_test.py seqs16nothr 4 30000 2>&1 | tee bench/burst_seqs16nothr.log
log "8-stream burst"
python3 -u burst_test.py seqs16nothrx8 8 30000 2>&1 | tee bench/burst_seqs16nothrx8.log
log "SEQS19 CHAIN COMPLETE"
