#!/usr/bin/env bash
# E-A9c (program note 5.32): the fairness control of E-A9. Does the same self-snap (carried state toward the centroid of the class the model itself predicted) also help the plain S5 GRU w256 and
# LSTM w208 (constant-lr and annealed)? The recurrence is unrolled by hand; 1,024 positions, n = 200. Evaluation only. One CPU thread, BelowNormal, idempotent (skips when the result file exists).
# Roughly 2-3 minutes per model.
#
# Usage: nohup bash scripts/run_ea9c.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Z=results/Z1-pilot
OUT=$Z/snap-recurrent
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ ! -f "$OUT.json" ]; then
  MODELS=()
  for m in perm5-gru-w256-s1-cur perm5-gru-w256-s1-cur-lrdecay0.3 perm5-lstm-w208-s1-cur perm5-lstm-w208-s1-cur-lrdecay0.3; do
    if [ -f "$Z/$m.pt" ]; then MODELS+=("$Z/$m.pt"); fi
  done
  log "BASLA E-A9c (CPU, 1 parcacik): ayni yapisma GRU / LSTM'e de yariyor mu (adalet kontrolu), ${#MODELS[@]} model, 1.024 konum n = 200, egitimsiz"
  $PY -B -m horizon.snap_recurrent "${MODELS[@]}" --out "$OUT.json" > "$OUT.txt" 2>&1
  if [ -f "$OUT.json" ]; then log "BITTI E-A9c (bkz $OUT.txt)"; else log "HATA E-A9c: sonuc dosyasi yok (bkz $OUT.txt)"; fi
fi
