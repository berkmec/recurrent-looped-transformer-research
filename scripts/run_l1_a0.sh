#!/usr/bin/env bash
# E-A0 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.41): the A0 twin of L1 - the same recipe, data order, initial
# weights and learning-rate schedule as results/L1-scratch-w928-s401, feedback OFF (mline.lm arm 'a0'), stopped at update 8,000 on
# the full 30,518-update schedule. GPU, about 0.7 s per update (measured), peak VRAM about 3.5 GB.
# Resumable: started again, mline.longrun goes on from its own latest.pt.
#
# Usage: nohup bash scripts/run_l1_a0.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."
export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
OUT=results/L1-a0-w928-s401
LOG=results/M1-finetune-distance/queue.log
CMD="python -B -m mline.longrun --out $OUT --recipe m2-draft --tokens 500000000 --arm a0 --stop-at 8000"
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
if [ -f "$OUT/stopped.json" ]; then log "ATLA E-A0 (stopped.json var)"; exit 0; fi
"$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
  E-A0-L1-a0-w928-s401 L1-a0-w928-s401 claude-4797b78c "$CMD" "notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.41" '{"seed": 401}' "$0" >> "$LOG" 2>&1 \
  || log "MANIFEST HATA E-A0"
log "BASLA E-A0 L1-a0-w928-s401 (GPU; geri beslemesiz ikiz; 8.000 guncelleme, program 30.518 uzerinden)"
"$PY" -B -m mline.longrun --out "$OUT" --recipe m2-draft --tokens 500000000 --arm a0 --stop-at 8000 >> "$OUT/train.log" 2>&1
if [ -f "$OUT/stopped.json" ]; then log "BITTI E-A0 (bkz $OUT/probe.jsonl)"; else log "HATA E-A0 (bkz $OUT/train.log)"; fi
