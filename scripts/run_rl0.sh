#!/usr/bin/env bash
# E-RL0 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.68): SFT then RL on the RLT paper's addition task for L1 (feedback on) and A0F
# (feedback off), both at 30,518 updates; CPU, one process per arm (2 threads, BelowNormal; the GPU stays with stage 1). L1 starts when the BLiMP
# queue has ended, A0F when the E-C-ROT queue has ended (free memory). Each arm first times a short smoke run on unregistered cells (no registered
# number is seen), then runs in full; every stage checkpoints, so a pause for the user is a kill and a relaunch. Then the registered reading.
# Usage: nohup bash scripts/run_rl0.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
OUT=results/RL0
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.68"
log() { echo "[$(date +%H:%M:%S)] [rl0] $*" >> "$LOG"; }
mkdir -p "$OUT"

"$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
  E-RL0-L1-A0F RL0 claude-4797b78c "python -B -m mline.rl0 run <L1 | A0F latest.pt> --out $OUT/<arm> --seed 1 --threads 2; python -B -m mline.rl0 reading" "$P" '{"seed": 1}' "$0" >> "$LOG" 2>&1 \
  || log "MANIFEST HATA E-RL0"

timing() {  # smoke directory -> seconds per SFT update and per RL step, projected full run
  "$PY" -B -c "
import json, sys
def per(f):
    rows = [json.loads(l) for l in open(f, encoding='utf8') if l.strip()]
    return (rows[-1]['seconds'] - rows[0]['seconds']) / max(1, len(rows) - 1)
s, r = per(sys.argv[1] + '/sft.jsonl'), per(sys.argv[1] + '/rl.jsonl')
print('SFT %.2f s/guncelleme, RL %.2f s/adim -> egitim ~%.1f sa (degerlendirmeler haric)' % (s, r, (2000 * s + 500 * r) / 3600))" "$1" 2>/dev/null
}

lane() {  # arm name, weights, the queue line to wait for
  local name="$1" weights="$2" pattern="$3"
  [ -f "$OUT/$name/final.json" ] && return 0
  log "E-RL0 $name bekliyor: '$pattern' satiri"
  until grep -q -E "$pattern" "$LOG"; do sleep 120; done
  if [ ! -f "$OUT/smoke-$name/final.json" ]; then
    "$PY" -B -m mline.rl0 run "$weights" --out "$OUT/smoke-$name" --seed 1 --threads 2 --smoke > "$OUT/smoke-$name.txt" 2>&1
    log "E-RL0 $name duman testi (kayit disi hucreler): $(timing "$OUT/smoke-$name")"
  fi
  log "BASLA E-RL0 $name (SFT 2.000 + RL 500; CPU, 2 is parcacigi)"
  "$PY" -B -m mline.rl0 run "$weights" --out "$OUT/$name" --seed 1 --threads 2 >> "$OUT/$name.txt" 2>&1
  if [ -f "$OUT/$name/final.json" ]; then log "BITTI E-RL0 $name"; else log "HATA E-RL0 $name (bkz $OUT/$name.txt)"; fi
}

lane L1 results/L1-scratch-w928-s401/latest.pt "\[blimp\] E-BLIMP kuyruk BITTI" &
lane A0F results/L1-a0-w928-s401/latest.pt "\[ecrot\] E-C-ROT kuyrugu BITTI" &
wait
if [ -f "$OUT/L1/final.json" ] && [ -f "$OUT/A0F/final.json" ]; then
  "$PY" -B -m mline.rl0 reading "$OUT/L1/final.json" "$OUT/A0F/final.json" > "$OUT/reading.json" 2> "$OUT/reading.err" \
    && log "BITTI E-RL0 okuma: $("$PY" -B -c "import json; r = json.load(open('$OUT/reading.json', encoding='utf8')); print('S1 L1', r['q1']['L1']['verdict'], 'delta %+.3f' % r['q1']['L1']['delta'], '| S2', r['q2']['verdict'], 'D %+.3f' % r['q2']['delta'], '| SFT LEARNED', r['sft_learned'])" 2>/dev/null)" \
    || log "HATA E-RL0 okuma (bkz $OUT/reading.err)"
fi
log "E-RL0 kuyruk BITTI"
