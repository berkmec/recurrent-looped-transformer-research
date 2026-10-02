#!/usr/bin/env bash
# E-A9k (program note 5.37): how rarely may the self-snap fire? Snap interval k in {32, 64, 96, 128, 256} (class selector: the class the model itself predicted), the constant-lr and the annealed S5 Core w128,
# n = 40 dev sequences of 8,192 positions, attention windowed to the last 64 positions as in E-A9L; the measured first-error hazard is compared with the renewal prediction from the E-A5 injection
# hazards (horizon/snap_interval.py). Evaluation only. One CPU thread, BelowNormal, idempotent (skips when the result file exists). Roughly 10-12 minutes per model.
#
# Usage: nohup bash scripts/run_ea9k.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Z=results/Z1-pilot
OUT=$Z/snap-interval
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ ! -f "$OUT.json" ]; then
  MODELS=()
  for m in perm5-core-w128-s1-cur perm5-core-w128-s1-cur-lrdecay0.3; do
    if [ -f "$Z/$m.pt" ]; then MODELS+=("$Z/$m.pt"); fi
  done
  log "BASLA E-A9k (CPU, 1 parcacik): yakalama araligi k doz-yaniti (32..256) ve yenilenme ongorusu, ${#MODELS[@]} S5 Core, 8.192 konum n = 40, egitimsiz"
  $PY -B -m horizon.snap_interval "${MODELS[@]}" --out "$OUT.json" > "$OUT.txt" 2>&1
  if [ -f "$OUT.json" ]; then log "BITTI E-A9k (bkz $OUT.txt)"; else log "HATA E-A9k: sonuc dosyasi yok (bkz $OUT.txt)"; fi
fi
