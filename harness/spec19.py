import bench_matrix as bm, json
specs = bm.gw("GET", "/admin/specs")
base = [s for s in specs if s["id"] == 17][0]
spec = json.loads(json.dumps(base["spec"])); spec["name"] = "glm-exl3-grouped-7168-seqs16-nothr"
spec["env"].update({"MAX_NUM_SEQS": "16", "EXTRA_ARGS": '--compilation-config {"max_cudagraph_capture_size":128}'})
have = [s for s in specs if s["name"] == spec["name"]]
body = {"name": spec["name"], "icon": "⚡", "priority": "elastic", "spec": spec}
sid = (bm.gw("PATCH", f"/admin/specs/{have[0]['id']}", body) or True) and have[0]["id"] if have else bm.gw("POST", "/admin/specs", body)["id"]
print("spec", sid, spec["name"], "->", spec["env"]["MAX_NUM_SEQS"], "seqs, EXTRA_ARGS:", spec["env"]["EXTRA_ARGS"])
