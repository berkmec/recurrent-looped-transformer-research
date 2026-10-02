#!/usr/bin/env bash
# E-A10p (program note 5.33): can the gated Hopfield cleanup MODULE (horizon/cleanup.py) clean when its prototypes are set from the data instead of learned? The module is added to the trained
# constant-lr and annealed S5 Core w128 (and, as the control, the S4 Core w32 / w64), prototypes = per-class (labelled) or k-means (label-free) means of r = RMSNorm(in-range state), temperature 50,
# gate 0.999. 1,024 positions, n = 200. Evaluation only. One CPU thread, BelowNormal, idempotent (skips when the result file exists). Roughly 3 minutes per model.
#
# Usage: nohup bash scripts/run_ea10p.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Z=results/Z1-pilot
OUT=$Z/cleanup-probe
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ ! -f "$OUT.json" ]; then
  MODELS=()
  for m in perm5-core-w128-s1-cur perm5-core-w128-s1-cur-lrdecay0.3 perm4-core-w32-s1-cur perm4-core-w64-s1-cur; do
    if [ -f "$Z/$m.pt" ]; then MODELS+=("$Z/$m.pt"); fi
  done
  log "BASLA E-A10p (CPU, 1 parcacik): Hopfield temizleme modulu, prototipleri veriden kurulunca temizleyebiliyor mu, ${#MODELS[@]} Core modeli, 1.024 konum n = 200, egitimsiz"
  $PY -B -m horizon.cleanup_probe "${MODELS[@]}" --out "$OUT.json" > "$OUT.txt" 2>&1
  if [ -f "$OUT.json" ]; then log "BITTI E-A10p (bkz $OUT.txt)"; else log "HATA E-A10p: sonuc dosyasi yok (bkz $OUT.txt)"; fi
fi
