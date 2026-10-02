#!/usr/bin/env bash
# E-P: is the constant error floor of the S_4 GRU and LSTM an optimisation artefact (a constant learning rate, never annealed) rather than a limit of the arm?
# Same data stream, curriculum, budget and seed as the stored seed-1 baselines (15,000 updates, lengths 1,2,3,4,6,8,12,16, lr 0.003, width 64) but the learning rate
# decays linearly to ~0 over the LAST 30 % of the updates (the baselines reached the final length-16 stage at step ~2,000, so the whole decay lies inside it).
# Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.10 (same thresholds as E-D; python -B -m horizon.readings ep).
# DEV split; NOT the frozen cycle. One CPU process, BelowNormal, idempotent.
#
# Usage: nohup bash scripts/run_ep.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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
for arm in gru lstm; do
  id="perm4-$arm-w64-s1-cur-lrdecay0.3"
  if [ -f "$OUT/$id.json" ]; then continue; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (CPU, 1 parcacik, 15000 adim, lr 0.003 son %30'da dogrusal sonume iner, mufredat $CUR)"
  $PY -B -m horizon.depth_run --task perm4 --arm "$arm" --width 64 --seed 1 --steps 15000 --lr 0.003 --curriculum "$CUR" --lr-decay-fraction 0.3 --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; fi
done
log "Z1 kuyruk EP BITTI"
