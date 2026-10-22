#!/usr/bin/env bash
# E-S-B2X (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.47): sk4 baselines at their own peak lr 1e-3 with a 2x budget
# (30,000 updates), seeds 1-2. Work-stealing: start TWO copies (two CPU processes). One thread, BelowNormal, RUNMAN manifest per run.
# Usage: nohup bash scripts/run_es_b2x.sh > /dev/null 2>&1 &   (twice; progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ES-b2x
LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] [b2x] $*" >> "$LOG"; }

job() {   # arm width seed
  local id="sk4-$1-w$2-s$3-lr0.001-30k"
  [ -f "$OUT/$id.json" ] && return 0
  mkdir "$OUT/$id.lock" 2>/dev/null || return 0
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "$id" ES-b2x claude-4797b78c "python -B -m horizon.sk_run --task sk4 --arm $1 --width $2 --seed $3 --lr 0.001 --steps 30000 --out $OUT/$id.json --log-every 1000" \
    "notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.47" "{\"seed\": $3}" "$0" >> "$LOG" 2>&1 || log "MANIFEST HATA $id"
  log "BASLA ES $id (CPU, 1 parcacik, 30000 adim, tepe lr 0.001)"
  "$PY" -B -m horizon.sk_run --task sk4 --arm "$1" --width "$2" --seed "$3" --lr 0.001 --steps 30000 --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI ES $id"; else log "HATA ES $id (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

job gru 128 1
job gru 128 2
job lstm 112 1
job lstm 112 2
job tfm 64 1
job tfm 64 2
log "kopya $$ listeyi bitirdi (b2x)"
