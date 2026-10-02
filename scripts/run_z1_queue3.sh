#!/usr/bin/env bash
# Z1 queue 3: close the matching gaps the first two queues left (DEV split, single seed; NOT the frozen cycle).
#   - S_4: LSTM at the Core's parameter count (width 112: 106,648 parameters; Core w64 98,328, GRU w128 105,240), same updates and curriculum;
#   - S_5: LSTM at about the Core's parameter count (width 208: 397,816; Core w128 409,720), the GRU w256 run's budget and learning rate;
#   - the NoPE transformer (4,000 updates did not reach the in-range threshold on the counter, 15,000 updates did not get past curriculum
#     length 6 on S_4): a longer budget for each, to tell "not trained yet" from "cannot".
# Run only when the other queue leaves a slot free (two experiment processes in total). Idempotent, BelowNormal, finished jobs are skipped.
#
# Usage: Z1_PAR=1 nohup bash scripts/run_z1_queue3.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OUT=results/Z1-pilot LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

job() {   # task arm width seed steps lr curriculum-or-"-" [tag]   (tag keeps a longer run of the same condition from overwriting the shorter one;
          # "-" and not an empty word: xargs below drops empty quotes and the tag would slide into the curriculum slot)
  local suffix="" extra=() cur="${7:--}" note=""
  if [ "$cur" != "-" ]; then suffix="-cur"; extra=(--curriculum "$cur"); note=", mufredat $cur"; fi
  local id="$1-$2-w$3-s$4$suffix${8:-}"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (CPU, 1 parcacik, $5 adim, lr $6$note)"
  $PY -B -m horizon.depth_run --task "$1" --arm "$2" --width "$3" --seed "$4" --steps "$5" --lr "$6" "${extra[@]}" --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; fi
}
export -f log job

CUR=1,2,3,4,6,8,12,16
printf '%s\n' \
  "job perm4 lstm 112 1 15000 0.003 $CUR" \
  "job perm5 lstm 208 1 30000 0.001 $CUR" \
  "job count tfm 64 1 16000 0.001 - -long" \
  "job perm4 tfm 64 1 60000 0.001 $CUR -long" \
  | xargs -P "${Z1_PAR:-1}" -I{} bash -c '{}'
log "Z1 kuyruk 3 BITTI"
