#!/usr/bin/env bash
# Z1 queue 2: the baseline arms the pilots still lack, run with the SAME budget and curriculum as the Core runs so the comparisons are matched:
#   - counter: LSTM (unbounded cell state: the recurrent baseline that can count) and the NoPE transformer;
#   - S_4: GRU with the Core's curriculum and budget (the earlier GRU run had 4,000 updates and no curriculum), GRU at the Core's parameter
#     count (width 128), LSTM, transformer.
# At most Z1_PAR (default 1) jobs at a time: start it only when the other queue leaves a slot free (two experiment processes in total).
# Idempotent (skips finished jobs), BelowNormal priority, DEV split only; NOT the frozen cycle.
#
# Usage: Z1_PAR=1 nohup bash scripts/run_z1_queue2.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OUT=results/Z1-pilot LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

job() {   # task arm width seed steps lr [curriculum]
  local suffix="" extra=()
  if [ -n "${7:-}" ]; then suffix="-cur"; extra=(--curriculum "$7"); fi
  local id="$1-$2-w$3-s$4$suffix"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (CPU, 1 parcacik, $5 adim, lr $6${7:+, mufredat $7})"
  $PY -B -m horizon.depth_run --task "$1" --arm "$2" --width "$3" --seed "$4" --steps "$5" --lr "$6" "${extra[@]}" --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; fi
}
export -f log job

CUR=1,2,3,4,6,8,12,16
printf '%s\n' \
  'job count lstm 64 1 4000 0.003' \
  "job perm4 gru 64 1 15000 0.003 $CUR" \
  'job count tfm 64 1 4000 0.001' \
  "job perm4 lstm 64 1 15000 0.003 $CUR" \
  "job perm4 gru 128 1 15000 0.003 $CUR" \
  "job perm4 tfm 64 1 15000 0.001 $CUR" \
  | xargs -P "${Z1_PAR:-1}" -I{} bash -c '{}'
log "Z1 kuyruk 2 BITTI"
