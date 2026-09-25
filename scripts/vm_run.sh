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
GRACE=${GRACE:-240}
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
    # A grace period rather than an immediate halt: the results have to be collected off the box before it goes,
    # and an unattended machine must still not idle for days. Cancel with: sudo shutdown -c
    echo "powering off in ${GRACE} minutes (cancel with: sudo shutdown -c)" | tee -a "$LOG"
    sudo shutdown -h +${GRACE} || true
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

# BOTH arms run here, on one machine. The first probe's padded numbers exist, but they were measured on another
# machine, GPU, operating system and torch version, so using them as the control would put four extra differences
# into a design that is meant to carry one. Packed runs first because it is the arm that might be positive, and
# an early answer is worth having; the control follows and settles attribution if it is.
say "M0 PACKED (treatment): does the needle capability appear at all?"
retry "M0 packed" "$PY" -B -u -m mline.emergence --out results/M0-packed --steps 5000 --batch 32 --every 250 \
    --needle-rate 0.125 --device cuda

say "M0 PADDED (control): the same run with the old padded mix, one term different"
retry "M0 padded" "$PY" -B -u -m mline.emergence --out results/M0-padded --steps 5000 --batch 32 --every 250 \
    --needle-rate 0.125 --device cuda --no-pack

say "paired comparison"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path


def curve(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding='utf8').strip().splitlines()]
    return {r['step']: r for r in rows if 'needle' in r}


a, b = curve('results/M0-padded/probe.jsonl'), curve('results/M0-packed/probe.jsonl')
print('%6s | %-24s | %-24s' % ('step', 'PADDED (control)', 'PACKED (treatment)'))
print('%6s | %8s %7s %7s | %8s %7s %7s' % ('', 'val bpb', 'tr@32', 'byte', 'val bpb', 'tr@32', 'byte'))
for step in sorted(set(a) & set(b)):
    ra, rb = a[step], b[step]
    print('%6d | %8.3f %7.3f %7.3f | %8.3f %7.3f %7.3f'
          % (step, ra['validation']['bits_per_byte'], ra['needle']['train@32']['exact_match'],
             ra['needle']['train@32']['byte_accuracy'], rb['validation']['bits_per_byte'],
             rb['needle']['train@32']['exact_match'], rb['needle']['train@32']['byte_accuracy']))
PY

# The gate: M1 is only worth 22 hours if the plain arm can do the task at all. The rule reads the probe's own
# record rather than a human impression of the curve - the train@32 set is the leading indicator, drawn from the
# keys and distances training actually shows.
say "probe verdict (read from the packed arm, which is the one M1 would inherit)"
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

say "calibration: batch size and whether the three arms should share the card"
# The 24 GB card re-opens two questions the 8 GB one settled by force. Measured, not assumed - this campaign has
# already generalised a utilisation figure from the toy models to the 30M one and been wrong about it.
"$PY" -B scripts/calibrate.py --device cuda 2>&1 | tee -a "$LOG"
PLAN=$("$PY" -c "import json;print(json.load(open('results/calibration.json'))['plan'])")
say "M1 execution plan: $PLAN"

say "M1: three arms, paired"
retry "M1 freeze" "$PY" -B -m mline.train --freeze --out results/M1

if [ "$PLAN" = "parallel" ]; then
  # One process per arm. Safe because run_one already claims a condition with active.lock and skips a finished
  # one via complete.json, so two workers cannot take the same arm. Each keeps its own resumable checkpoint.
  # The idents come FROM the protocol, never typed here: a hardcoded name drifts silently the moment the width
  # or the seed changes, and a worker would then be launched for a condition that does not exist.
  IDENTS=$("$PY" -c "from mline import train as t; print(' '.join(t.ident(c) for c in t.conditions(t.protocol())))")
  pids=()
  for ident in $IDENTS; do
    say "starting $ident"
    "$PY" -B -u -m mline.train --run-all --out results/M1 --device cuda --only "$ident" >>"$LOG" 2>&1 &
    pids+=($!)
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [ "$failed" = 1 ]; then
    say "a worker stopped; re-running serially resumes each arm from its own checkpoint"
    retry "M1 run (recovery)" "$PY" -B -u -m mline.train --run-all --out results/M1 --device cuda
  fi
else
  retry "M1 run" "$PY" -B -u -m mline.train --run-all --out results/M1 --device cuda
fi

retry "M1 analysis" "$PY" -B -m mline.analyze --out results/M1

say "all done"
