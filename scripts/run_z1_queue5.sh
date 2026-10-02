#!/usr/bin/env bash
# Z1 queue 5: seeds 2 and 3 of the matched S_4 comparison. The central pilot reading (Core: no first error in 203,200 sequence-steps; GRU and LSTM:
# a constant per-step error of about 1e-4 .. 8e-4, also at the Core's parameter count) rests on ONE training seed per arm; the per-step error rate
# of a trained model depends on the particular weights, so the evaluation-sampling statistics in the master document do not cover it.
# Same budget and curriculum as the seed-1 runs (15,000 updates, lengths 1,2,3,4,6,8,12,16, lr 0.003): Core w64, GRU w64 / w128, LSTM w64 / w112.
# DEV split; NOT the frozen cycle. Cheap recurrent runs first (minutes), the Core runs (about 35 minutes each) last.
#
# At most Z1_PAR (default 1) jobs at a time; start it only when the other queues leave a slot free (two experiment processes in total).
# Idempotent (skips finished jobs), BelowNormal priority.
#
# Usage: Z1_PAR=1 nohup bash scripts/run_z1_queue5.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OUT=results/Z1-pilot LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

job() {   # task arm width seed steps lr curriculum
  local id="$1-$2-w$3-s$4-cur"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (CPU, 1 parcacik, $5 adim, lr $6, mufredat $7)"
  $PY -B -m horizon.depth_run --task "$1" --arm "$2" --width "$3" --seed "$4" --steps "$5" --lr "$6" --curriculum "$7" --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; fi
}
export -f log job

CUR=1,2,3,4,6,8,12,16
JOBS=()
for seed in 2 3; do
  JOBS+=("job perm4 lstm 64 $seed 15000 0.003 $CUR" "job perm4 gru 64 $seed 15000 0.003 $CUR"
         "job perm4 lstm 112 $seed 15000 0.003 $CUR" "job perm4 gru 128 $seed 15000 0.003 $CUR")
done
for seed in 2 3; do JOBS+=("job perm4 core 64 $seed 15000 0.003 $CUR"); done
printf '%s\n' "${JOBS[@]}" | xargs -P "${Z1_PAR:-1}" -I{} bash -c '{}'
log "Z1 kuyruk 5 BITTI"
