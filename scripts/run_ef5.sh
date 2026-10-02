#!/usr/bin/env bash
# E-F5: do the Core's feedback gain alpha or its PEAK learning rate decide the S5 solution family (E-P2: the break is not lr annealing; E-M5: the margin erodes)? S5 Core w128, seed 1,
# 30,000 updates, lr annealed to ~0 over the last 30 %, curriculum 1..16, exactly the E-P2 run except for ONE change per run: alpha 1.0 (the sealed Core's gain is 0.1; other groups used 1),
# peak lr 1e-3 (the GRU / LSTM value; the Core used 3e-3), alpha 0.3. Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.24
# (python -B -m horizon.readings ef5). DEV split; NOT the frozen cycle. One CPU process at a time (about 30-75 minutes per run), BelowNormal, idempotent, lock directory per job.
#
# Usage: nohup bash scripts/run_ef5.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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
job() {   # suffix lr alpha
  local id="perm5-core-w128-s1-cur-lrdecay0.3-$1"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (E-F5; CPU, 1 parcacik, 30000 adim, tepe lr $2, alfa $3, lr son %30'da ~0'a, mufredat $CUR)"
  $PY -B -m horizon.depth_run --task perm5 --arm core --width 128 --seed 1 --steps 30000 --lr "$2" --alpha "$3" --curriculum "$CUR" --lr-decay-fraction 0.3 \
    --out "$OUT/$id.json" --log-every 2000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

job alpha1.0 0.003 1.0
job lr0.001 0.001 0.1
job alpha0.3 0.003 0.3
log "E-F5 kuyruk BITTI"
