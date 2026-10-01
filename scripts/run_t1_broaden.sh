#!/usr/bin/env bash
# T1h: does key matching learned on a NARROW key alphabet turn into a general comparison when the alphabet is then BROADENED?
#
# T1r-b (small Core, single-fact start + MQAR with replacement, 17 key letters, 3000 updates) matches keys on its training letters
# (first digit 0.98 on every question) and not at all on the never-seen T-Z (0.27-0.39). T1g trains the same thing on the broad
# alphabet from the single-fact start and has not made the jump yet. Here the jump is already made, on 17 letters; the question is
# whether continuing on 53 symbols (T-Z still never in a key) makes it general - a narrow-then-broad curriculum.
#
#   T1h  T1r-b's weights, MQAR with replacement, key length 4, BROAD alphabet, 2000 updates.
#
# Usage: nohup bash scripts/run_t1_broaden.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
ROOT=results/M1-finetune-distance
LOG=$ROOT/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }
name=T1h-t1rb-then-broad
steps=2000
out=$ROOT/$name
mkdir -p "$out"
if [ ! -f "$out/README.txt" ]; then
  cat > "$out/README.txt" <<README
EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. A small sealed Core (width 128), CPU, one thread.
T1h: T1r-b's weights (key matching on 17 letters, first digit 0.98 on training letters, 0.27-0.39 on T-Z), then MQAR with
replacement, key length 4, over the BROAD alphabet (53 symbols; T-Z never appear in a training key), $steps updates.
Pre-registered reading (written before any step ran), held_k4_first in the eval rows (the T-Z score):
- GENERALISES: held_k4_first >= 0.6 at some eval step.
- if train_k4_first stays high while held_k4_first stays near 0.33, a narrow-then-broad curriculum does not make matching general
  at this size and budget.
Recipe: python -B -m mline.mqar_scratch --out $out --steps $steps --facts 4 --queries 4 --key-length 4 --key-alphabet broad --seed 331 --with-replacement --init-from $ROOT/T1r-b-single-then-replace/latest.pt
README
fi
if [ "$(lines "$out/probe.jsonl")" -lt "$steps" ]; then
  log "BASLA $name (CPU, 1 parcacik, $steps adim)"
  $PY -B -m mline.mqar_scratch --out "$out" --steps "$steps" --facts 4 --queries 4 --key-length 4 --key-alphabet broad \
      --train-filler 0,64 --seed 331 --with-replacement --init-from "$ROOT/T1r-b-single-then-replace/latest.pt" \
      >> "$out/train.log" 2>&1
fi
if [ "$(lines "$out/probe.jsonl")" -lt "$steps" ]; then log "HATA $name: egitim bitmedi ($(lines "$out/probe.jsonl") adim)"; exit 1; fi
log "BITTI $name"
