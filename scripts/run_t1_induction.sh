#!/usr/bin/env bash
# T1d / T1e: does teaching the match-then-copy circuit on rows that cannot be solved by position make key-addressed retrieval
# (MQAR) learnable - where a single-fact start (T1s -> T1b) produced a position-locked model and MQAR from scratch (T1a) has
# plateaued at chance?
#
#   T1d  small Core (width 128) from scratch on induction rows only: a random string, filler, the same string again.
#   T1e  T1d's weights, then exactly T1a's MQAR training (same seed, same batches). T1a vs T1b vs T1e is the comparison.
#
# CPU only, one thread per job. Resumable and idempotent like scripts/run_t1.sh. Progress goes to queue.log.
# Usage: nohup bash scripts/run_t1_induction.sh > /dev/null 2>&1 &
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

job T1d-scratch-induction 2000 \
    "T1d: small Core from scratch on induction rows only (random string, filler, the same string; no key/value template, no positional shortcut). Does the match-then-copy circuit form, and how fast?" \
    "- the circuit FORMS: copy_accuracy (exact next byte on the second copy after its first 2 bytes; chance ~0.016) >= 0.9 at some eval step.
- no rise by 2000 steps is a statement about budget, width or learning rate, not about the architecture." \
    -- --source induction --seg-length 8-32 --gap 0,96 --lead 0,16 --skip 2 --seed 321
job T1e-induction-then-mqar 3000 \
    "T1e: T1d's weights, then exactly T1a's MQAR training (same seed, same batches). Does an induction start make key-addressed retrieval learnable where MQAR from scratch (T1a) and a single-fact start (T1b) were not?" \
    "- induction start HELPS: T1e reaches decoy-asked share >= 0.5 on TRAIN_KEYS/4 at an earlier eval step than T1a (or T1a never does), and HELD_KEYS/4 >= 0.6 at the end.
- otherwise it is neutral or worse than T1a." \
    -- --facts 4 --queries 4 --key-length 1-4 --key-alphabet wide --train-filler 0,64 --seed 301 \
       --init-from "$ROOT/T1d-scratch-induction/latest.pt"
log "T1 induction BITTI"
