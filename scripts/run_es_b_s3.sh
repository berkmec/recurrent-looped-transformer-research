#!/usr/bin/env bash
# E-S stage B, seed 3 for the two arms the advantage rests on (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.55): the
# feedback Core w64 and its no-feedback twin core_nofb w64, sk4, the stage-B settings unchanged (15,000 updates, lr 3e-3 annealed over
# the last 30 %). Two CPU processes in parallel, one thread each, BelowNormal, RUNMAN manifest per run.
# Usage: nohup bash scripts/run_es_b_s3.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ES-pilot
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] [es3] $*" >> "$LOG"; }

job() {   # arm width
  local id="sk4-$1-w$2-s3"
  [ -f "$OUT/$id.json" ] && return 0
  mkdir "$OUT/$id.lock" 2>/dev/null || return 0
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "$id" ES-pilot claude-4797b78c "python -B -m horizon.sk_run --task sk4 --arm $1 --width $2 --seed 3 --steps 15000 --out $OUT/$id.json --log-every 1000" \
    "notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.55" '{"seed": 3}' "$0" >> "$LOG" 2>&1 || log "MANIFEST HATA $id"
  log "BASLA ES $id (CPU, 1 parcacik, 15000 adim, asama B ayarlari)"
  "$PY" -B -m horizon.sk_run --task sk4 --arm "$1" --width "$2" --seed 3 --steps 15000 --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI ES $id"; else log "HATA ES $id (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

job core 64 & job core_nofb 64 & wait
log "ES tohum 3 BITTI"
