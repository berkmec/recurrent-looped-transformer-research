#!/usr/bin/env bash
# Run the M-line queue on a rented box, survive spot preemption, and switch the meter off at the end.
#
# ORDER MATTERS AND IS NOT NEGOTIABLE. The packed probe runs FIRST, because M1 costs 22 hours and would measure
# nothing if the needle capability still does not appear - that is exactly what the padded probe demonstrated at
# a cost of 9 hours. The probe answers a yes/no question; M1 only makes sense on a yes.
#
# Usage, on the VM:
#   bash vm_run.sh /data/rlt                 # probe, then M1 if the probe earns it
#   bash vm_run.sh /data/rlt --no-shutdown   # leave the box up (costs money; use only while debugging)
set -euo pipefail

ROOT="${1:-/data/rlt}"
SHUTDOWN=1
[ "${2:-}" = "--no-shutdown" ] && SHUTDOWN=0

REPO="$ROOT/2026-09-18-phase2"
PY="$ROOT/venv/bin/python"
export MLINE_DATA="$ROOT/mline-data/cnn-bytes"
cd "$REPO"

LOG="$REPO/results/vm-run-$(date -u +%Y%m%dT%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

say() { printf '\n== %s == %s\n' "$1" "$(date -u +%H:%M:%S)" | tee -a "$LOG"; }
finish() {
  code=$?
  say "run ended with status $code"
  if [ "$SHUTDOWN" = 1 ]; then
    echo "powering off in 60s (cancel with: sudo shutdown -c)" | tee -a "$LOG"
    sudo shutdown -h +1 || true
  fi
  exit $code
}
trap finish EXIT

# Spot instances get taken back. Every run here checkpoints with its optimizer moments and step, and resuming is
# bit-exact (tests/test_m1_resume.py compares a resumed run against an uninterrupted one weight for weight), so a
# preemption costs at most one checkpoint interval. Re-running this script after a restart continues where it
# stopped; it does not start over.
retry() {
  local label="$1"; shift
  local attempt=1
  until "$@" >>"$LOG" 2>&1; do
    if [ $attempt -ge 5 ]; then say "$label failed $attempt times, stopping"; return 1; fi
    say "$label interrupted (attempt $attempt), resuming from its last checkpoint"
    attempt=$((attempt + 1))
    sleep 20
  done
  say "$label complete"
}

say "step cost on THIS card"
"$PY" -B -m mline.train --probe 2>&1 | tee -a "$LOG"

say "M0 packed probe: does the needle capability appear at all?"
retry "M0 probe" "$PY" -B -u -m mline.emergence --out results/M0-packed --steps 5000 --batch 32 --every 250 \
    --needle-rate 0.125 --device cuda

# The gate: M1 is only worth 22 hours if the plain arm can do the task at all. The rule reads the probe's own
# record rather than a human impression of the curve - the train@32 set is the leading indicator, drawn from the
# keys and distances training actually shows.
say "probe verdict"
if ! "$PY" - <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
rows = [json.loads(l) for l in Path('results/M0-packed/probe.jsonl').read_text(encoding='utf8').strip().splitlines()]
m = [r for r in rows if 'needle' in r]
best = max((r['needle']['train@32']['exact_match'] for r in m), default=0.)
byte = max((r['needle']['train@32']['byte_accuracy'] for r in m), default=0.)
print(f'best train@32 exact {best:.4f}  byte {byte:.4f}  over {len(m)} measurements')
ok = best > 0.02 or byte > 0.25
print('VERDICT:', 'capability present - M1 is worth running' if ok else
      'still absent - M1 would measure nothing; stopping for a human decision')
sys.exit(0 if ok else 1)
PY
then
  say "stopping before M1 on purpose"
  exit 0
fi

say "M1: three arms, paired"
retry "M1 freeze" "$PY" -B -m mline.train --freeze --out results/M1
retry "M1 run"    "$PY" -B -u -m mline.train --run-all --out results/M1 --device cuda
retry "M1 analysis" "$PY" -B -m mline.analyze --out results/M1

say "all done"
