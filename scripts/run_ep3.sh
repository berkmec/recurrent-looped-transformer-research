#!/usr/bin/env bash
# E-P3: how far must the learning rate be annealed to remove the S4 GRU/LSTM error floor? (E-P: to ~0 over the last 30 % -> 0 / 203,200 and 1 / 203,200 first errors.)
# The L1 long run anneals only to 0.1 x the base rate (mline/longrun.py lr_at, floor 0.1), so this sets the schedule floor of the next long run.
# Same data stream, curriculum, budget and seed as the stored constant-lr and E-P runs; the rate decays linearly over the LAST 30 % to f x the base rate,
# f in {0.3, 0.1, 0.03, 0.01} (the final rates 9e-4, 3e-4, 9e-5, 3e-5). Seed 1, S4, width 64, GRU and LSTM, 15,000 updates.
# Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.12 (python -B -m horizon.readings ep3). DEV; NOT the frozen cycle.
# One CPU process, BelowNormal, idempotent.
#
# Usage: nohup bash scripts/run_ep3.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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
for floor in 0.1 0.01 0.3 0.03; do
  for arm in gru lstm; do
    id="perm4-$arm-w64-s1-cur-lrfloor$floor"
    if [ -f "$OUT/$id.json" ]; then continue; fi
    echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
    log "BASLA Z1 $id (CPU, 1 parcacik, 15000 adim, lr 0.003 son %30'da dogrusal $floor x lr'ye iner, mufredat $CUR)"
    $PY -B -m horizon.depth_run --task perm4 --arm "$arm" --width 64 --seed 1 --steps 15000 --lr 0.003 --curriculum "$CUR" --lr-decay-fraction 0.3 --lr-floor "$floor" --out "$OUT/$id.json" --log-every 2000 >> "$OUT/$id.log" 2>&1
    if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; fi
  done
done
log "Z1 kuyruk EP3 BITTI"
