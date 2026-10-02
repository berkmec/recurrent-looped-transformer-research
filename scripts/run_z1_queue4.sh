#!/usr/bin/env bash
# Z1 queue 4: the counter with a larger trained peak P (count16 / count32). Every recurrent arm so far learned the same saturating counter at
# P = 8; does extrapolation appear once the finite-automaton solution (P + 1 states) gets expensive? Same arms, width 64, lr 0.003, no curriculum;
# training length 8 P; evaluation at 1, 2, 4 ... times the trained peak (capped at 512). DEV split, single seed; NOT the frozen cycle.
#
#   core       the Core at P = 16 (about 1.4 s per update: 8,000 updates take about three hours). P = 32 waits for this result.
#   recurrent  GRU and LSTM at P = 16 and 32 (minutes to an hour each).
#
# At most Z1_PAR (default 1) jobs at a time; start it only when the other queues leave a slot free (two experiment processes in total).
# Idempotent (skips finished jobs), BelowNormal priority.
#
# Usage: Z1_PAR=1 nohup bash scripts/run_z1_queue4.sh core > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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

case "${1:-}" in
  core)      JOBS=('job count16 core 64 1 8000 0.003') ;;
  recurrent) JOBS=('job count16 gru 64 1 8000 0.003' 'job count16 lstm 64 1 8000 0.003'
                   'job count32 gru 64 1 12000 0.003' 'job count32 lstm 64 1 12000 0.003') ;;
  *) echo "usage: run_z1_queue4.sh core|recurrent" >&2; exit 1 ;;
esac
printf '%s\n' "${JOBS[@]}" | xargs -P "${Z1_PAR:-1}" -I{} bash -c '{}'
log "Z1 kuyruk 4 ($1) BITTI"
