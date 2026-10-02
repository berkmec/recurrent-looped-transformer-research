#!/usr/bin/env bash
# E-K5: does WIDTH (state dimension per class) remove the S5 Core's long-position margin erosion (E-M5, note 5.20.1)? S5 Core at widths 256 (state dim / classes = 2.13, the S5 GRU w256's ratio)
# and 192 (1.6), seed 1, 30,000 updates, lr 0.003 annealed to ~0 over the last 30 % (exactly the E-P2 recipe; the lr is NOT rescaled with the width: width is the only change).
# Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.22 (python -B -m horizon.readings ek5). DEV split; NOT the frozen cycle.
# One CPU process at a time (about 1.5-2.5 hours per run), BelowNormal, idempotent; an atomic lock directory per job lets two queue chains share it.
#
# Usage: nohup bash scripts/run_ek5.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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
job() {   # width
  local id="perm5-core-w$1-s1-cur-lrdecay0.3"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (E-K5; CPU, 1 parcacik, 30000 adim, lr 0.003 son %30'da ~0'a, genislik $1, mufredat $CUR)"
  $PY -B -m horizon.depth_run --task perm5 --arm core --width "$1" --seed 1 --steps 30000 --lr 0.003 --curriculum "$CUR" --lr-decay-fraction 0.3 \
    --out "$OUT/$id.json" --log-every 2000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

job 256
job 192
log "E-K5 kuyruk BITTI"
