#!/usr/bin/env bash
# E-RM0R8 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.72): the toy write-load test, arms ds32 / ds128 / ds128c, seeds 1 and
# 2, 6,000 updates each; two at a time (the user is at the computer: at most two CPU jobs), one thread each, BelowNormal, unbuffered logs,
# RUNMAN manifests; then the registered reading per seed. An arm whose result file exists is skipped, so a relaunch after
# scripts/pause_for_user.ps1 redoes only the unfinished arms (they keep no checkpoint: about 0.3 h for ds32, 1 h for ds128).
# Usage: nohup bash scripts/run_erm_r8.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ERM-R8
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.72"
log() { echo "[$(date +%H:%M:%S)] [erm8] $*" >> "$LOG"; }
mkdir -p "$OUT"

run() {    # arm seed
  local id="$1-s$2"
  if [ -f "$OUT/$id.json" ]; then return; fi
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "E-RM0R8-$id" ERM-R8 claude-4797b78c "python -B -m horizon.erm_r8 train --arm $1 --seed $2 --steps 6000 --out $OUT/$id.json" "$P" "{\"seed\": $2}" "$0" >> "$LOG" 2>&1 \
    || log "MANIFEST HATA $id"
  log "BASLA E-RM0R8 $id"
  "$PY" -B -u -m horizon.erm_r8 train --arm "$1" --seed "$2" --steps 6000 --out "$OUT/$id.json" > "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI E-RM0R8 $id"; else log "HATA E-RM0R8 $id (bkz $OUT/$id.log)"; fi
}
verdict() {   # seed -> the registered verdict of that seed's three arms
  "$PY" -B -m horizon.erm_r8 reading "$OUT/ds32-s$1.json" "$OUT/ds128-s$1.json" "$OUT/ds128c-s$1.json" 2>/dev/null \
    | "$PY" -B -c "import json, sys; r = json.load(sys.stdin); print(r['verdict'], r['arms'], {k: round(v, 3) for k, v in r['acc_d3'].items()})" 2>/dev/null
}

run ds32 1 & run ds128 1 & wait
run ds128c 1 & run ds32 2 & wait
run ds128 2 & run ds128c 2 & wait
log "E-RM0R8 okuma tohum 1: $(verdict 1)"
log "E-RM0R8 okuma tohum 2: $(verdict 2)"
log "E-RM0R8 kuyruk BITTI"
