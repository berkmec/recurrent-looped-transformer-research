#!/usr/bin/env bash
# E-J0 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.43): the exactness gate of the parallel fixed-point evaluation of the
# feedback loop, on L1's last checkpoint; no training. Shares the GPU with E-A0F, so this process is capped at 40 % of the GPU's
# memory (it fails before it can starve the training run).
# Usage: nohup bash scripts/run_ej0.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."
export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
OUT=results/EJ0-jacobi
LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] [ej0] $*" >> "$LOG"; }
[ -f "$OUT/exactness.json" ] && { log "ATLA E-J0 (exactness.json var)"; exit 0; }
CMD="python -B -m mline.jacobi --weights results/L1-scratch-w928-s401/latest.pt --out $OUT/exactness.json (GPU memory fraction 0.4)"
"$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
  E-J0-exactness EJ0-jacobi claude-4797b78c "$CMD" "notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.43" '{"seed": 401}' "$0" >> "$LOG" 2>&1 \
  || log "MANIFEST HATA E-J0"
log "BASLA E-J0 (egitimsiz; GPU, bellek tavani %40)"
"$PY" -B -c "import sys, torch; torch.cuda.set_per_process_memory_fraction(0.4); sys.argv = ['mline.jacobi', '--weights', 'results/L1-scratch-w928-s401/latest.pt', '--out', '$OUT/exactness.json']; from mline import jacobi; jacobi.main()" \
  > "$OUT/exactness.txt" 2>&1
if [ -f "$OUT/exactness.json" ]; then log "BITTI E-J0: $("$PY" -B -c "import json; r=json.load(open('$OUT/exactness.json'))['reading']; print(r['verdict'], 'K_y', r['passes_forward'], 'K_g', r['passes_gradient'])" 2>/dev/null)"; else log "HATA E-J0 (bkz $OUT/exactness.txt)"; fi
