#!/usr/bin/env bash
# E-A9d (program note 5.38, descriptive): the snap interval k in {1, 2, 4, 8, 16, 32, 64, 128} for the S5 Core (constant-lr, annealed) and the S5 GRU w256 / LSTM w208 (constant-lr, annealed) on ONE
# protocol (n = 200 dev sequences x 1,024 positions, class selector); k_max per model. Evaluation only. One CPU thread, BelowNormal, idempotent. Roughly 5 minutes in total.
#
# Usage: nohup bash scripts/run_ea9d.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Z=results/Z1-pilot
OUT=$Z/snap-grid
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ ! -f "$OUT.json" ]; then
  MODELS=()
  for m in perm5-core-w128-s1-cur perm5-core-w128-s1-cur-lrdecay0.3 perm5-gru-w256-s1-cur perm5-gru-w256-s1-cur-lrdecay0.3 perm5-lstm-w208-s1-cur perm5-lstm-w208-s1-cur-lrdecay0.3; do
    if [ -f "$Z/$m.pt" ]; then MODELS+=("$Z/$m.pt"); fi
  done
  log "BASLA E-A9d (CPU, 1 parcacik): yakalama araligi izgarasi Core vs GRU/LSTM, ${#MODELS[@]} model, 1.024 konum n = 200, egitimsiz, betimleyici"
  $PY -B -m horizon.snap_grid "${MODELS[@]}" --out "$OUT.json" > "$OUT.txt" 2>&1
  if [ -f "$OUT.json" ]; then log "BITTI E-A9d (bkz $OUT.txt)"; else log "HATA E-A9d: sonuc dosyasi yok (bkz $OUT.txt)"; fi
fi
