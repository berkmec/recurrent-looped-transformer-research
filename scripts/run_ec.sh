#!/usr/bin/env bash
# E-C: scale-diverse excursions for the counter. The label saturates at 8 (nine classes) but TRAINING climbs to a log-uniform depth per sequence
# (8 .. R_max), so every excursion has to be counted; the test goes past R_max. Reading rule written before this ran:
# notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.3. GRU and LSTM, width 64, seed 1, 12,000 updates, lr 0.003, no curriculum. DEV split; NOT the
# frozen cycle. First count8_32 (R_max 32, length 256; LSTM about 22 minutes, GRU about 43), then count8_64 (R_max 64, length 512; about twice that).
#
# At most Z1_PAR (default 1) jobs at a time; start it only when the other queues leave a slot free (two experiment processes in total).
# Idempotent (skips finished jobs), BelowNormal priority.
#
# Usage: Z1_PAR=1 nohup bash scripts/run_ec.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OUT=results/Z1-pilot LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

job() {   # task arm width seed steps lr points
  local id="$1-$2-w$3-s$4"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  # an atomic lock directory: two queue chains may both call this script (07:55: one chain after the S4 seeds, one after E-P4), and each job must run once
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (CPU, 1 parcacik, $5 adim, lr $6, noktalar $7)"
  $PY -B -m horizon.depth_run --task "$1" --arm "$2" --width "$3" --seed "$4" --steps "$5" --lr "$6" --points "$7" --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}
export -f log job

POINTS=8,16,32,64,128,256,512
JOBS=("job count8_32 lstm 64 1 12000 0.003 $POINTS" "job count8_32 gru 64 1 12000 0.003 $POINTS"
      "job count8_64 lstm 64 1 12000 0.003 $POINTS" "job count8_64 gru 64 1 12000 0.003 $POINTS")
printf '%s\n' "${JOBS[@]}" | xargs -P "${Z1_PAR:-1}" -I{} bash -c '{}'
log "Z1 kuyruk EC BITTI"
