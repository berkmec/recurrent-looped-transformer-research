#!/usr/bin/env bash
# T8/M-3b: tohum-ayristirma 3x3 (perm4/gru/w64/15k, wseed{1,2,3} x dseed{101,102,103}, eval_seed 999).
# Sealed pre-reg: notes/_hive_M3b_TASARIM.md (commit 8857f99). ~0,5 CPU-h envelope.
# Single lane, BelowNormal, idempotent (finished jobs skipped; atomic lock dir per job).
# Step 0 (acceptance, run once before queue): --seed 1 --wseed 1 --dseed 1 --eval-seed 1
#   --n-eval 100 must reproduce results/Z1-pilot/perm4-gru-w64-s1-cur.json bit-for-bit
#   (minus wall/train_seconds); if not: STOP, record, do not queue.
# Usage: nohup bash scripts/run_m3b.sh > /dev/null 2>&1 &   (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/M3b
LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

CUR=1,2,3,4,6,8,12,16
job() {   # wseed dseed
  local id="perm4-gru-w64-w$1d$2"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA M3b $id (CPU, 1 parcacik, 15000 adim, mufredat $CUR, wseed $1, dseed $2, eval 999)"
  $PY -B -m horizon.depth_run --task perm4 --arm gru --width 64 --seed "$1" --wseed "$1" --dseed "$2" --eval-seed 999 --steps 15000 --lr 0.003 --curriculum "$CUR" --n-eval 200 --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI M3b $id"; else log "HATA M3b $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

for w in 1 2 3; do
  for d in 101 102 103; do
    job "$w" "$d"
  done
done
log "M3b kuyruk BITTI"
