#!/usr/bin/env bash
# T1g: the small-Core analogue of D5 - does a BROAD key alphabet (53 symbols, T-Z never in a training key) plus questions drawn
# WITH replacement make key matching generalise to letters it never saw?
#
# Why this is a fair preview: T1b (small Core, single-fact start, MQAR) reached first-question accuracy 0.84 on its training
# letters after 3000 updates and 0.38 on T-Z - the same two facts as D2 (30M): key matching is learned, and it is letter-specific.
# The small Core is a cheap (CPU, ~75 min) test bed for what would make it general.
#
#   T1g  T1s's weights (the single-fact start that sped T1b up), MQAR with replacement, key length 4, broad alphabet, 4000 updates.
#
# Usage: nohup bash scripts/run_t1_broad.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
ROOT=results/M1-finetune-distance
LOG=$ROOT/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }
name=T1g-single-then-broad-replace
steps=4000
out=$ROOT/$name
mkdir -p "$out"
if [ ! -f "$out/README.txt" ]; then
  cat > "$out/README.txt" <<README
EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. A small sealed Core (width 128), CPU, one thread.
T1g: the small-Core analogue of D5. T1s's weights, then MQAR (4 facts / 4 questions, questions drawn WITH replacement) with key
length 4 over the BROAD alphabet (53 symbols; T-Z never appear in a training key), $steps updates.
Pre-registered reading (written before any step ran), on the training-shaped score in every eval row:
- key matching LEARNED: train_k4_first >= 0.8.
- GENERALISES: held_k4_first >= 0.6 at that step (T-Z were never in a key). T1b without the broad alphabet reached 0.93 / 0.57
  (training letters / T-Z, T-Z being elimination only), so a held-out score well above 0.57 would be the new result.
- if train_k4_first rises but held_k4_first stays near T1b's, a broad alphabet alone does not make matching general.
Recipe: python -B -m mline.mqar_scratch --out $out --steps $steps --facts 4 --queries 4 --key-length 4 --key-alphabet broad --seed 311 --with-replacement --init-from $ROOT/T1s-scratch-single/latest.pt
README
fi
if [ "$(lines "$out/probe.jsonl")" -lt "$steps" ]; then
  log "BASLA $name (CPU, 1 parcacik, $steps adim)"
  $PY -B -m mline.mqar_scratch --out "$out" --steps "$steps" --facts 4 --queries 4 --key-length 4 --key-alphabet broad \
      --train-filler 0,64 --seed 311 --with-replacement --init-from "$ROOT/T1s-scratch-single/latest.pt" >> "$out/train.log" 2>&1
fi
if [ "$(lines "$out/probe.jsonl")" -lt "$steps" ]; then log "HATA $name: egitim bitmedi ($(lines "$out/probe.jsonl") adim)"; exit 1; fi
log "BITTI $name"
