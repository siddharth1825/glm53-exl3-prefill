#!/usr/bin/env bash
# Orchestrates the rest of step 0 + the grouped gate, unattended:
#   wait 4d -> logprob capture (stock PR77 path) -> bench 4f (grouped) -> logprob capture (grouped) -> compare -> bench 4e (legacy control)
cd ~/glm-opt
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -qE "PR77 BENCH COMPLETE|Traceback|never became" bench_pr77_4d.log; do sleep 20; done
grep -qE "Traceback|never became" bench_pr77_4d.log && { log "4d failed; stopping chain"; exit 1; }
log "4d done; capturing logprobs on the stock (PR77) path while deployment is still up"
python3 logprob_capture.py stock77 2>&1 | tail -4
log "starting 4f (grouped prefill)"
python3 -u bench_pr77.py 4f > bench_pr77_4f.log 2>&1
tail -3 bench_pr77_4f.log
if grep -q "PR77 BENCH COMPLETE" bench_pr77_4f.log; then
  log "capturing logprobs on the grouped path"
  python3 logprob_capture.py grouped 2>&1 | tail -4
  log "=== numerics gate: stock77 vs grouped ==="
  python3 logprob_capture.py --compare stock77 grouped
else
  log "4f did not complete; see bench_pr77_4f.log"
fi
log "starting 4e (legacy control)"
python3 -u bench_pr77.py 4e > bench_pr77_4e.log 2>&1
tail -2 bench_pr77_4e.log
log "CHAIN COMPLETE"
