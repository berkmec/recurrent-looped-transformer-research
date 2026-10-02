#!/usr/bin/env bash
# E-A9L (program note 5.29): how long does the self-snapped S5 Core stay exact? The constant-lr and the lr-annealed S5 Core w128, snap of the carried state toward the centroid of the class the
# model itself predicted (every 16 steps: primary; every step; nearest centroid every 16 steps), dev sequences of 32,768 positions (n = 40, all conditions) and 131,072 positions (n = 8, the
# primary condition only), encoder / cross-attention restricted to the last 64 positions (E-A5: no effect on these models; it keeps the cost banded). Evaluation only. One CPU thread,
# BelowNormal, idempotent (skips when the result file exists). Roughly 15-20 minutes per model.
#
# Usage: nohup bash scripts/run_ea9l.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Z=results/Z1-pilot
OUT=$Z/snap-long
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ ! -f "$OUT.json" ]; then
  MODELS=()
  for m in perm5-core-w128-s1-cur perm5-core-w128-s1-cur-lrdecay0.3; do
    if [ -f "$Z/$m.pt" ]; then MODELS+=("$Z/$m.pt"); fi
  done
  log "BASLA E-A9L (CPU, 1 parcacik): kendi-sinifina yapisan S5 Core cok uzun dizilerde (32.768 n = 40; 131.072 n = 8), ${#MODELS[@]} model, egitimsiz"
  $PY -B -m horizon.snap_long "${MODELS[@]}" --lengths 32768,131072 --n 40,8 --window 64 --out "$OUT.json" > "$OUT.txt" 2>&1
  if [ -f "$OUT.json" ]; then log "BITTI E-A9L (bkz $OUT.txt)"; else log "HATA E-A9L: sonuc dosyasi yok (bkz $OUT.txt)"; fi
fi
