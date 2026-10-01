#!/usr/bin/env bash
# T1r: T1a / T1b again with the questions drawn WITH replacement - no elimination shortcut.
#
# T1a and T1b (no replacement) ended with a loss that fell to 0.28-0.35 and a first-digit score on the FIRST question of only
# 0.33-0.42 - but exactly 1.00 on the LAST question: with 4 facts asked once each, the last question is answered by elimination
# (the one fact not asked yet), and they had learned that instead of matching keys. Here every asked fact is drawn independently,
# so each question needs its key. Same small Core (width 128), same seed and batches as T1a/T1b apart from the question draw.
#
#   T1r-a  from scratch.       T1r-b  from T1s's weights (the single-fact, position-locked start).
#
# CPU only, one thread per job, below-normal priority is set by the caller. Resumable and idempotent. Progress: queue.log.
# Usage: nohup bash scripts/run_t1_replace.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
ROOT=results/M1-finetune-distance
LOG=$ROOT/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }

job() {   # name steps purpose reading -- <mqar_scratch args>
  local name=$1 steps=$2 purpose=$3 reading=$4
  shift 5
  local out=$ROOT/$name
  mkdir -p "$out"
  if [ ! -f "$out/README.txt" ]; then
    cat > "$out/README.txt" <<README
EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. A small sealed Core (width 128), CPU, one thread.
$purpose
Pre-registered reading (written before any step ran), on the FIRST question (no elimination is possible there) of the
training-shaped score mqar_eval writes into every eval row (train_k4_first etc.; chance for a first digit is 0.1, copying a
random fact scores about 0.33):
$reading
Recipe: python -B -m mline.mqar_scratch --out $out --steps $steps $*
README
  fi
  if [ "$(lines "$out/probe.jsonl")" -lt "$steps" ]; then
    log "BASLA $name (CPU, 1 parcacik, $steps adim)"
    $PY -B -m mline.mqar_scratch --out "$out" --steps "$steps" "$@" >> "$out/train.log" 2>&1
  fi
  if [ "$(lines "$out/probe.jsonl")" -lt "$steps" ]; then log "HATA $name: egitim bitmedi ($(lines "$out/probe.jsonl") adim)"; return 1; fi
  log "BITTI $name"
}

READ="- key matching LEARNED: train_k4_first >= 0.8 at some eval step (with replacement every question needs the key).
- GENERALISES to unseen letters: held_k4_first >= 0.6 at the same step (T-Z never appear in a training key).
- if it stays near 0.33 the budget or width is too small, or the architecture cannot do it: that is the question this run asks."

job T1r-a-scratch-replace 3000 \
    "T1r-a: small Core from scratch on MQAR with the questions drawn WITH replacement. Can this architecture learn key-addressed retrieval once elimination is not available?" \
    "$READ" \
    -- --facts 4 --queries 4 --key-length 1-4 --key-alphabet wide --train-filler 0,64 --seed 301 --with-replacement &
JA=$!
job T1r-b-single-then-replace 3000 \
    "T1r-b: T1s's weights (single-fact, position-locked), then the same MQAR training with replacement. Does a position-locked start slow or block key matching?" \
    "$READ
- single-fact start HINDERS: T1r-b's train_k4_first stays >= 0.3 below T1r-a's at the last step, or reaches 0.5 at least twice as late." \
    -- --facts 4 --queries 4 --key-length 1-4 --key-alphabet wide --train-filler 0,64 --seed 301 --with-replacement \
       --init-from "$ROOT/T1s-scratch-single/latest.pt" &
JB=$!
wait $JA
wait $JB
log "T1r BITTI"
