#!/usr/bin/env bash
# E-D5: does training with carried-state noise remove the S5 Core's long-position break that lr annealing could not (E-P2, note 5.11.1)? S5 Core w128, seed 1, 30,000 updates, lr 0.003
# annealed to ~0 over the last 30 % (exactly the E-P2 run) PLUS relative Gaussian noise (sigma 0.1 and 0.2 times the state rms) on the carried state at every step but the first during TRAINING
# only; evaluation is clean. Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.21 (python -B -m horizon.readings ed5). DEV split; NOT the frozen cycle.
# One CPU process at a time (about 75-90 minutes per run), BelowNormal, idempotent; an atomic lock directory per job lets two queue chains share it.
#
# Usage: nohup bash scripts/run_ed5.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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
job() {   # sigma
  local id="perm5-core-w128-s1-cur-lrdecay0.3-noise$1"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (E-D5; CPU, 1 parcacik, 30000 adim, lr 0.003 son %30'da ~0'a, durum gurultusu sigma $1, mufredat $CUR)"
  $PY -B -m horizon.depth_run --task perm5 --arm core --width 128 --seed 1 --steps 30000 --lr 0.003 --curriculum "$CUR" --lr-decay-fraction 0.3 --state-noise "$1" \
    --out "$OUT/$id.json" --log-every 2000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

job 0.2
job 0.1
log "E-D5 kuyruk BITTI"
