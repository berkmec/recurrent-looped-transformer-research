#!/usr/bin/env bash
# E-C0: is the saturating counter of the default design universal over training seeds? The pilots (GRU, LSTM, Core: first descent identical to the third decimal,
# right only at the floor arrival) rest on ONE training seed per arm; the literature's best-seed counters extrapolate 4-10x more than the typical one.
# Seeds 2-8 of the default counter (peak 8, length 64, 4,000 updates, lr 0.003, width 64) for GRU and LSTM (seed 1 exists), plus E-M: the Core without
# feedback (its parallel path: minutes) as a control for an attention-ratio route to counting. Reading rule written before this ran:
# notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.6 (python -B -m horizon.readings ec0). DEV split; NOT the frozen cycle.
#
# At most Z1_PAR (default 1) jobs at a time; start it only when the other queues leave a slot free (two experiment processes in total).
# Idempotent (skips finished jobs), BelowNormal priority.
#
# Usage: Z1_PAR=1 nohup bash scripts/run_ec0.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OUT=results/Z1-pilot LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

job() {   # task arm width seed steps lr
  local id="$1-$2-w$3-s$4"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (CPU, 1 parcacik, $5 adim, lr $6)"
  $PY -B -m horizon.depth_run --task "$1" --arm "$2" --width "$3" --seed "$4" --steps "$5" --lr "$6" --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; fi
}
export -f log job

JOBS=("job count core_nofb 64 1 4000 0.003")
for seed in 2 3 4 5 6 7 8; do
  JOBS+=("job count lstm 64 $seed 4000 0.003" "job count gru 64 $seed 4000 0.003")
done
printf '%s\n' "${JOBS[@]}" | xargs -P "${Z1_PAR:-1}" -I{} bash -c '{}'
log "Z1 kuyruk EC0 BITTI"
