import bench_matrix as bm, datetime as dt, sys
mins = int(sys.argv[1]) if len(sys.argv) > 1 else 75
rows = bm.gw("GET", "/admin/logs?limit=300")
rows = rows.get("items", rows) if isinstance(rows, dict) else rows
now = dt.datetime.now(dt.timezone.utc)
def ts(r):
    s = r.get("ts") or r.get("created_at") or ""
    try: return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception: return None
sel = []
for r in rows:
    t = ts(r)
    if not t or (now - t).total_seconds() > mins * 60: continue
    if "GLM" not in (r.get("model") or ""): continue
    sel.append((t, r))
sel.sort(key=lambda x: x[0])
if rows and not sel: print("fields:", sorted(rows[0].keys()))
print(f"{len(sel)} GLM requests in the last {mins} min")
print(f"{'key':10s} {'IST':8s} {'prompt':>7s} {'compl':>6s} {'TTFT s':>7s} {'total s':>8s} {'prefill/s':>9s} {'dec/s':>6s}  status")
tot_p = tot_c = 0
for t, r in sel:
    p = r.get("prompt_tokens") or 0; c = r.get("completion_tokens") or 0
    ttft = (r.get("ttft_ms") or 0) / 1000; lat = (r.get("latency_ms") or 0) / 1000
    pre = p / ttft if ttft else 0; dec = c / (lat - ttft) if lat > ttft and c else 0
    tot_p += p; tot_c += c
    key = str(r.get("key_name") or r.get("key") or "")[:10]
    print(f"{key:10s} {(t + dt.timedelta(hours=5, minutes=30)).strftime('%H:%M:%S')} {p:7d} {c:6d} {ttft:7.2f} {lat:8.1f} {pre:9.0f} {dec:6.1f}  {r.get('status') or r.get('status_code') or ''} {r.get('finish_reason') or ''}")
print(f"totals: prompt {tot_p} tokens, completion {tot_c} tokens")
