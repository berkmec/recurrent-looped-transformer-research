#!/usr/bin/env bash
# E-A5x (program note 5.26.1, descriptive): how does the carried state's deviation from the canonical class state grow with its AGE since a clean injection (systematic vs random part)?
# Evaluation only, five Core models (three S5, two S4). One CPU thread, BelowNormal, idempotent (skips when the result file exists). Roughly 2 minutes per model.
#
# Usage: nohup bash scripts/run_ea5x.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Z=results/Z1-pilot
OUT=$Z/drift-profile
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ ! -f "$OUT.json" ]; then
  MODELS=()
  for m in perm5-core-w128-s1-cur perm5-core-w128-s1-cur-lrdecay0.3 perm5-core-w128-s1-cur-lrdecay0.3-noise0.2 perm4-core-w32-s1-cur perm4-core-w64-s1-cur; do
    if [ -f "$Z/$m.pt" ]; then MODELS+=("$Z/$m.pt"); fi
  done
  log "BASLA E-A5x (CPU, 1 parcacik): durum sapmasinin yasa gore buyumesi (sistematik / rastgele), ${#MODELS[@]} Core modeli, egitimsiz, betimleyici"
  $PY -B -m horizon.drift_profile "${MODELS[@]}" --out "$OUT.json" > "$OUT.txt" 2>&1
  if [ -f "$OUT.json" ]; then log "BITTI E-A5x (bkz $OUT.txt)"; else log "HATA E-A5x: sonuc dosyasi yok (bkz $OUT.txt)"; fi
fi
