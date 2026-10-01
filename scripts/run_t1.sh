#!/usr/bin/env bash
# T1: is key-addressed retrieval something the Core can learn at all, and does a single-fact start get in the way?
# CPU only, one thread per job (the GPU belongs to the decoy queue; scripts/_gpu_busy.ps1 ignores these jobs).
#
#   T1s  small Core (width 128, 0.44 M parameters) from scratch on the SINGLE-fact needle task (one fact, 4-letter keys A-M,
#        filler 0-64) - a stand-in for what every 30M checkpoint learned first.
#   T1a  the same small Core from scratch on MQAR (4 facts / 4 questions, key length 1-4, wide key alphabet).
#   T1b  T1s's weights, then exactly T1a's MQAR training (same seed => same batches as T1a). T1a vs T1b is the comparison.
#
# T1s and T1a run together; T1b starts when T1s ends. Resumable: each job continues from its own latest.pt, finished jobs are
# skipped. Progress goes to results/M1-finetune-distance/queue.log.
#
# Usage: nohup bash scripts/run_t1.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
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
Pre-registered reading (written before any step ran):
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

READ_MQAR="- key matching LEARNED: asked-share >= 0.8 for BOTH asks on TRAIN_KEYS/4 at some eval step.
- GENERALISES to unseen letters: the same on HELD_KEYS/4 (T-Z never appear in a training key), threshold 0.6.
- If it never leaves the position-locked pattern in the given steps, that is a statement about budget and width, not a verdict on the architecture."

job T1s-scratch-single 1500 \
    "T1s: small Core from scratch on the single-fact needle task only - what every 30M checkpoint learned first. Its decoy probe says whether it is position-locked." \
    "- position-locked: first-asked share near 0 and decoy-asked share near 1 at 4-letter keys (it copies the nearest fact), as the 30M plain model did." \
    -- --facts 1 --queries 1 --key-length 4 --key-alphabet default --train-filler 0,64 --seed 311 \
       --eval-alphabets TRAIN_KEYS,EVAL_KEYS --eval-key-lengths 4 &
T1S_PID=$!
job T1a-scratch-mqar 3000 \
    "T1a: small Core from scratch on MQAR only (4 facts / 4 questions, key length 1-4, wide key alphabet A-S). Can this architecture learn key-addressed retrieval?" \
    "$READ_MQAR" \
    -- --facts 4 --queries 4 --key-length 1-4 --key-alphabet wide --train-filler 0,64 --seed 301 &
T1A_PID=$!
wait $T1S_PID
job T1b-single-then-mqar 3000 \
    "T1b: T1s's weights, then exactly T1a's MQAR training (same seed, same batches). Does a single-fact start slow or block key matching?" \
    "$READ_MQAR
- single-fact start HINDERS: at the last step T1b's decoy-asked share on TRAIN_KEYS/4 stays >= 0.3 below T1a's, or T1b reaches 0.5 at least twice as late." \
    -- --facts 4 --queries 4 --key-length 1-4 --key-alphabet wide --train-filler 0,64 --seed 301 \
       --init-from "$ROOT/T1s-scratch-single/latest.pt"
wait $T1A_PID
log "T1 BITTI"
