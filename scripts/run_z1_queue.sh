#!/usr/bin/env bash
# Z1 queue: AT MOST TWO jobs at a time (xargs -P 2), in priority order; the next job starts only when one of the two slots is free.
# Every job is idempotent (skips when its result file exists), so killing and restarting this script loses at most the jobs in flight.
# Replaces run_z1_pilot*.sh, which opened five to seven workers at once and froze the machine (2026-10-02, recorded in the master document).
# Runs at BelowNormal priority (children inherit it). L1 (mline.longrun) and T1 (mline.mqar_scratch) are not part of this queue.
#
# Usage: nohup bash scripts/run_z1_queue.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

# lower our own priority first: processes created afterwards inherit BelowNormal
WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OUT=results/Z1-pilot LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT" results/Z1-G
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

gru_lane() {   # lane lanes
  log "BASLA Z1-G gru_control serit $1/$2"
  $PY -B -m horizon.gru_control --out results/Z1-G --lane "$1" --lanes "$2" >> "results/Z1-G/lane$1.log" 2>&1
  log "BITTI Z1-G serit $1/$2 (cikis kodu $?)"
}
export -f log job gru_lane

CUR=1,2,3,4,6,8,12,16
printf '%s\n' \
  'gru_lane 0 2' \
  'gru_lane 1 2' \
  'job count core 64 1 4000 0.003' \
  'job count gru 64 1 4000 0.003' \
  "job perm4 core 64 1 15000 0.003 $CUR" \
  "job perm5 gru 256 1 30000 0.001 $CUR" \
  "job perm5 core 128 1 30000 0.003 $CUR" \
  | xargs -P 2 -I{} bash -c '{}'
log "Z1 kuyruk BITTI"
