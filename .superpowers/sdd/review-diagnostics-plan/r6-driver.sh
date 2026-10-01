#!/usr/bin/env bash
# R6 main driver: freeze -> validate -> 20-condition ladder -> diagnostic. Stops on first launcher failure; OOM is a retained condition outcome, not a failure.
cd /e/rlt-research/2026-09-17-diagnostics || exit 1
PY="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
OUT=results/review-diagnostics-r6
LOG=.superpowers/sdd/review-diagnostics-plan/r6-driver.log
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "DRIVER START pid=$$"
log "STAGE FREEZE"; "$PY" -B -m review_cycle.memory_bench freeze --device cuda --out "$OUT" >> "$LOG" 2>&1 || { log "FAILED FREEZE"; exit 2; }
log "STAGE VALIDATE"; "$PY" -B -m review_cycle.memory_bench validate --out "$OUT" >> "$LOG" 2>&1 || { log "FAILED VALIDATE"; exit 3; }
for length in 32 128 512 1024; do
  for arm in full detach checkpoint segment segment_encoder_checkpoint; do
    log "STAGE RUN $arm L$length"
    "$PY" -B -m review_cycle.memory_bench run --out "$OUT" --length $length --arm $arm >> "$LOG" 2>&1 || { log "FAILED RUN $arm L$length"; exit 4; }
    log "DONE RUN $arm L$length status=$("$PY" -c "import json;print(json.load(open('$OUT/measurements/$arm-L$length/result.json'))['status'])" 2>/dev/null)"
  done
done
log "STAGE DIAGNOSTIC segment L128"; "$PY" -B -m review_cycle.memory_bench run --out "$OUT" --length 128 --arm segment --diagnostic >> "$LOG" 2>&1 || { log "FAILED DIAGNOSTIC"; exit 5; }
log "ALL DONE"
