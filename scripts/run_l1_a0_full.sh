#!/usr/bin/env bash
# E-A0F (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.42 a; the registered use of the 5.41 HELPS verdict): the A0 twin
# of L1 resumed from its own update-8,000 checkpoint to the end of the same schedule (30,518 updates). Nothing else changes; the
# 8,000-update verdict of 5.41 is final. GPU, about 0.66 s per update (measured), about 3.6 GB VRAM.
#
# Usage: nohup bash scripts/run_l1_a0_full.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."
export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
OUT=results/L1-a0-w928-s401
LOG=results/M1-finetune-distance/queue.log
CMD="python -B -m mline.longrun --out $OUT --recipe m2-draft --tokens 500000000 --arm a0"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
if [ -f "$OUT/complete.json" ]; then log "ATLA E-A0F (complete.json var)"; exit 0; fi
"$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
  E-A0F-L1-a0-w928-s401 L1-a0-w928-s401 claude-4797b78c "$CMD" "notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.42 a" '{"seed": 401}' "$0" >> "$LOG" 2>&1 \
  || log "MANIFEST HATA E-A0F"
log "BASLA E-A0F L1-a0-w928-s401 (GPU; 8.000 -> 30.518 guncelleme, ayni program)"
"$PY" -B -m mline.longrun --out "$OUT" --recipe m2-draft --tokens 500000000 --arm a0 >> "$OUT/train.log" 2>&1
if [ -f "$OUT/complete.json" ]; then log "BITTI E-A0F (bkz $OUT/probe.jsonl)"; else log "HATA E-A0F (bkz $OUT/train.log)"; fi
