#!/usr/bin/env bash
# E-P2: does the learning-rate decay that removed the S4 GRU/LSTM error floor (E-P: GRU 0 / 203,200 first errors, LSTM 1 / 203,200, against 7.6e-4 and 5.3e-4 with a
# constant rate) also remove the S5 floor / the S5 Core's break past position 128? And do the S4 results replicate on seeds 2 and 3?
# Same data stream, curriculum, budget and seed as the stored constant-lr runs; ONLY the learning rate differs: it decays linearly to ~0 over the LAST 30 % of the updates.
# Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.11 (python -B -m horizon.readings ep5 | epseeds).
# DEV split; NOT the frozen cycle. One CPU process at a time, BelowNormal, idempotent (skips finished jobs).
#
# Order (most informative first): S5 Core w128 (~90 min), S5 GRU w256 (~30 min), S5 LSTM w208 (~25 min), then the S4 replicates (4 x ~3 min).
# Usage: nohup bash scripts/run_ep2.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

CUR=1,2,3,4,6,8,12,16
job() {   # task arm width seed steps lr
  local id="$1-$2-w$3-s$4-cur-lrdecay0.3"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (CPU, 1 parcacik, $5 adim, lr $6 son %30'da dogrusal sonume iner, mufredat $CUR)"
  $PY -B -m horizon.depth_run --task "$1" --arm "$2" --width "$3" --seed "$4" --steps "$5" --lr "$6" --curriculum "$CUR" --lr-decay-fraction 0.3 --out "$OUT/$id.json" --log-every 2000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; fi
}

job perm5 core 128 1 30000 0.003
job perm5 gru 256 1 30000 0.001
job perm5 lstm 208 1 30000 0.001
for seed in 2 3; do
  job perm4 gru 64 "$seed" 15000 0.003
  job perm4 lstm 64 "$seed" 15000 0.003
done
log "Z1 kuyruk EP2 BITTI"
