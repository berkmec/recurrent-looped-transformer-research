#!/usr/bin/env bash
# E-A9b (program note 5.30): does the snap work with NO label at all? The prototypes are k-means centroids of the model's own in-range carried states (k = number of classes, k-means++
# initialisation, fixed seed); the selector is the nearest centroid in state space (no readout, no label). Five Core models (three S5, two S4 control), 1,024 positions, n = 200. Evaluation only.
# One CPU thread, BelowNormal, idempotent (skips when the result file exists). Roughly 3-4 minutes per model.
#
# Usage: nohup bash scripts/run_ea9b.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Z=results/Z1-pilot
OUT=$Z/snap-labelfree
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ ! -f "$OUT.json" ]; then
  MODELS=()
  for m in perm5-core-w128-s1-cur perm5-core-w128-s1-cur-lrdecay0.3 perm5-core-w128-s1-cur-lrdecay0.3-noise0.2 perm4-core-w32-s1-cur perm4-core-w64-s1-cur; do
    if [ -f "$Z/$m.pt" ]; then MODELS+=("$Z/$m.pt"); fi
  done
  log "BASLA E-A9b (CPU, 1 parcacik): etiketsiz yapisma (k-means prototipleri), ${#MODELS[@]} Core modeli, 1.024 konum n = 200, egitimsiz"
  $PY -B -m horizon.snap_labelfree "${MODELS[@]}" --out "$OUT.json" > "$OUT.txt" 2>&1
  if [ -f "$OUT.json" ]; then log "BITTI E-A9b (bkz $OUT.txt)"; else log "HATA E-A9b: sonuc dosyasi yok (bkz $OUT.txt)"; fi
fi
