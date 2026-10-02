#!/usr/bin/env bash
# E-A5 (program note 5.26): where does the S5 Core's long-position erosion live: in the non-recurrent context paths (A: attention windows) or in the history carried by the state
# (B: oracle re-injection of the true class's centroid state)? Evaluation only: three S5 Core models carry the verdict, two S4 Core models are the control. One CPU thread, BelowNormal,
# idempotent (skips when the result file exists). Roughly 7 minutes per model.
#
# Usage: nohup bash scripts/run_ea5.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Z=results/Z1-pilot
OUT=$Z/erosion-source
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ ! -f "$OUT.json" ]; then
  MODELS=()
  for m in perm5-core-w128-s1-cur perm5-core-w128-s1-cur-lrdecay0.3 perm5-core-w128-s1-cur-lrdecay0.3-noise0.2 perm4-core-w32-s1-cur perm4-core-w64-s1-cur; do
    if [ -f "$Z/$m.pt" ]; then MODELS+=("$Z/$m.pt"); fi
  done
  log "BASLA E-A5 (CPU, 1 parcacik): dikkat pencereleri (A) ve kahin durum yeniden enjeksiyonu (B), ${#MODELS[@]} Core modeli, 1.024 konum, n = 200, egitimsiz"
  $PY -B -m horizon.erosion_source "${MODELS[@]}" --out "$OUT.json" > "$OUT.txt" 2>&1
  if [ -f "$OUT.json" ]; then log "BITTI E-A5 (bkz $OUT.txt)"; else log "HATA E-A5: sonuc dosyasi yok (bkz $OUT.txt)"; fi
fi
