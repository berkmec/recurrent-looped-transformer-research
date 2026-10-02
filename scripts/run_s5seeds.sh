#!/usr/bin/env bash
# E-R5 (program note 5.31): is the S5 Core's long-position break a property of the architecture and recipe, or of seed 1? S5 Core w128, seeds 2 and 3, 30,000 updates, lr 0.003 CONSTANT,
# curriculum 1..16 (exactly the recipe of the stored seed-1 baseline `perm5-core-w128-s1-cur`, hazard 1.41e-3). Reading rule written before this ran: note 5.31
# (python -B -m horizon.readings es5seeds). DEV split; NOT the frozen cycle. One CPU process at a time (about 75-90 minutes per run), BelowNormal, idempotent; an atomic lock directory per
# job lets two queue chains share it.
#
# Usage: nohup bash scripts/run_s5seeds.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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
job() {   # seed
  local id="perm5-core-w128-s$1-cur"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (E-R5; CPU, 1 parcacik, 30000 adim, lr 0.003 sabit, genislik 128, tohum $1, mufredat $CUR)"
  $PY -B -m horizon.depth_run --task perm5 --arm core --width 128 --seed "$1" --steps 30000 --lr 0.003 --curriculum "$CUR" \
    --out "$OUT/$id.json" --log-every 2000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

job 2
job 3
log "E-R5 kuyruk BITTI"
