#!/usr/bin/env bash
# T1w: does a WIDER Core form the key-matching circuit in fewer updates? (the cheap evidence for "should we scale the model up")
#
# T1r-b (width 128, 0.44 M parameters) went: single-fact start for 1500 updates, then MQAR with replacement; first-digit score on
# its training letters 0.39 at 2000 updates and 0.96 at 2500. If more width makes that jump come earlier, scaling helps this recipe's
# data efficiency; if the jump does not move, parameters are not what is limiting key matching.
#
# Usage: nohup bash scripts/run_t1_width.sh WIDTH > /dev/null 2>&1 &      (heads = WIDTH / 32, as at width 128)
set -u
cd "$(dirname "$0")/.."
W=${1:?width}
H=$((W / 32))
PY="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
ROOT=results/M1-finetune-distance
LOG=$ROOT/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }

stage() {   # name steps readme -- args
  local name=$1 steps=$2 readme=$3
  shift 4
  local out=$ROOT/$name
  mkdir -p "$out"
  if [ ! -f "$out/README.txt" ]; then
    printf '%s\n' "EXPLORATORY DIAGNOSTIC, NOT A FROZEN CYCLE. One seed. Small sealed Core, width $W, heads $H, CPU, one thread." "$readme" \
        "Pre-registered reading: the update at which train_k4_first first reaches 0.8 (T1r-b at width 128: between 2000 and 2500 MQAR updates after a 1500-update single-fact start)." \
        "Recipe: python -B -m mline.mqar_scratch --out $out --steps $steps --width $W --heads $H $*" > "$out/README.txt"
  fi
  if [ "$(lines "$out/probe.jsonl")" -lt "$steps" ]; then
    log "BASLA $name (CPU, 1 parcacik, $steps adim)"
    $PY -B -m mline.mqar_scratch --out "$out" --steps "$steps" --width "$W" --heads "$H" "$@" >> "$out/train.log" 2>&1
  fi
  if [ "$(lines "$out/probe.jsonl")" -lt "$steps" ]; then log "HATA $name: egitim bitmedi ($(lines "$out/probe.jsonl") adim)"; return 1; fi
  log "BITTI $name"
}

stage "T1w$W-s" 1500 "Stage 1: single-fact needle task from scratch (the position-locked start)." -- \
    --facts 1 --queries 1 --key-length 4 --key-alphabet default --train-filler 0,64 --seed 311 \
    --eval-alphabets TRAIN_KEYS,EVAL_KEYS --eval-key-lengths 4 || exit 1
stage "T1w$W-r" 4000 "Stage 2: MQAR with replacement over the wide key alphabet, from stage 1's weights." -- \
    --facts 4 --queries 4 --key-length 1-4 --key-alphabet wide --train-filler 0,64 --seed 301 --with-replacement \
    --init-from "$ROOT/T1w$W-s/latest.pt"
log "T1w$W BITTI"
