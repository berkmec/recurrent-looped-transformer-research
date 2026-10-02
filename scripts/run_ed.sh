#!/usr/bin/env bash
# E-D: does training with relative Gaussian noise on the carried state lower the per-step error floor of the S_4 GRU and LSTM?
# Margin argument (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 4.2): hazard = erfc(m / sqrt 2), so a modest margin gain is a large hazard drop.
# Reading rule written before this ran: section 5.2 of that note. Same budget and curriculum as the seed-1 baselines (15,000 updates, lengths 1,2,3,4,6,8,12,16,
# lr 0.003, width 64); evaluation is CLEAN (no noise). DEV split; NOT the frozen cycle. Order: sigma 0.1 first (both arms), then 0.05 and 0.2.
#
# At most Z1_PAR (default 1) jobs at a time; start it only when the other queues leave a slot free (two experiment processes in total).
# Idempotent (skips finished jobs), BelowNormal priority.
#
# Usage: Z1_PAR=1 nohup bash scripts/run_ed.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OUT=results/Z1-pilot LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

job() {   # task arm width seed steps lr curriculum sigma
  local id="$1-$2-w$3-s$4-cur-noise$8"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (CPU, 1 parcacik, $5 adim, lr $6, mufredat $7, egitim-durum-gurultusu $8)"
  $PY -B -m horizon.depth_run --task "$1" --arm "$2" --width "$3" --seed "$4" --steps "$5" --lr "$6" --curriculum "$7" --state-noise "$8" --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; fi
}
export -f log job

CUR=1,2,3,4,6,8,12,16
JOBS=()
for sigma in 0.1 0.05 0.2; do
  for arm in gru lstm; do JOBS+=("job perm4 $arm 64 1 15000 0.003 $CUR $sigma"); done
done
printf '%s\n' "${JOBS[@]}" | xargs -P "${Z1_PAR:-1}" -I{} bash -c '{}'
log "Z1 kuyruk ED BITTI"
