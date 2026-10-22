#!/usr/bin/env bash
# E-L1C (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.50): L1 (feedback on, RLT-1) resumed from its own update-8,500
# checkpoint to the end of its schedule (30,518 updates) with CUDA graphs (E-J1: equivalent, 1.24x). Nothing else changes. GPU,
# about 30 hours. Reading at the end: python -B -m mline.a0_reading --end   (A0F vs L1 at 12,000 ... 30,518)
# Usage: nohup bash scripts/run_l1_cont.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."
export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
OUT=results/L1-scratch-w928-s401
LOG=results/M1-finetune-distance/queue.log
CMD="python -B -m mline.longrun --out $OUT --recipe m2-draft --tokens 500000000 --graphs"
log() { echo "[$(date +%H:%M:%S)] [l1c] $*" >> "$LOG"; }
if [ -f "$OUT/complete.json" ]; then log "ATLA E-L1C (complete.json var)"; exit 0; fi
"$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
  E-L1C-L1-scratch-w928-s401 L1-scratch-w928-s401 claude-4797b78c "$CMD" "notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.50" '{"seed": 401}' "$0" >> "$LOG" 2>&1 \
  || log "MANIFEST HATA E-L1C"
log "BASLA E-L1C L1 8.500 -> 30.518 (GPU, CUDA graph)"
"$PY" -B -m mline.longrun --out "$OUT" --recipe m2-draft --tokens 500000000 --graphs >> "$OUT/train.log" 2>&1
if [ -f "$OUT/complete.json" ]; then
  log "BITTI E-L1C"
  "$PY" -B -m mline.a0_reading --end --out "$OUT/reading-end.json" > /dev/null 2>&1 \
    && log "BITTI E-L1C okuma: $("$PY" -B -c "import json; r=json.load(open('$OUT/reading-end.json')); print(r['verdict'], 'delta_end', r.get('delta_last'))" 2>/dev/null)" \
    || log "HATA E-L1C okuma"
else
  log "HATA E-L1C (bkz $OUT/train.log)"
fi
