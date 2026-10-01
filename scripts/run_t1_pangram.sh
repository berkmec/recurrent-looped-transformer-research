#!/usr/bin/env bash
# T1p: the FAIR small-Core test of "does key matching generalise to letters that were never keys?"
#
# In T1b/T1g/T1h/T1r the held-out letters T-Z (U-Z in particular) appear nowhere in the training text, so their embeddings are still
# the random values the model was created with: that matching does not work on them is almost by construction. The 30M models were
# trained on language, where all letters are trained, and still failed (D2, D3) - that result stands. Here the filler is a PANGRAM
# (every letter, never in a key), so T-Z are trained letters that were never keys: the same situation as in the 30M model.
#
#   T1s2  small Core from scratch, single-fact needle task, pangram filler (the position-locked start that sped T1b/T1r-b up).
#   T1p   T1s2's weights, MQAR with replacement, key length 4, BROAD alphabet (53 symbols, T-Z never a key), pangram filler.
#
# Usage: nohup bash scripts/run_t1_pangram.sh > /dev/null 2>&1 &
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

job T1s2-scratch-single-pangram 1500 \
    "T1s2: T1s again (small Core from scratch, single-fact needle task) but with a PANGRAM as filler, so every letter is a trained letter." \
    "- position-locked like T1s (first-asked share near 0, decoy-asked near 1 on the decoy probe); only the start for T1p." \
    -- --facts 1 --queries 1 --key-length 4 --key-alphabet default --train-filler 0,64 --seed 311 --filler-kind pangram \
       --eval-alphabets TRAIN_KEYS,EVAL_KEYS --eval-key-lengths 4
job T1p-single-then-broad-pangram 4000 \
    "T1p: T1s2's weights, then MQAR with replacement, key length 4, the BROAD alphabet (53 symbols; T-Z never a key) and pangram filler. The fair small-Core version of the generalisation test: T-Z are trained letters that were never keys." \
    "- key matching LEARNED: train_k4_first >= 0.8 (T1g, the same without the pangram, reached 0.73 at 2500 updates).
- GENERALISES: held_k4_first >= 0.6 at the same step. A held-out score near 0.33 here is NOT explained by untrained embeddings." \
    -- --facts 4 --queries 4 --key-length 4 --key-alphabet broad --train-filler 0,64 --seed 311 --with-replacement \
       --filler-kind pangram --init-from "$ROOT/T1s2-scratch-single-pangram/latest.pt"
log "T1p BITTI"
