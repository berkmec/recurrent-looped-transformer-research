#!/usr/bin/env bash
# Takeover queue: the decoy-aware fine-tunes first, then the (idempotent) control queue finishes whatever is left.
#
# Why first: mline/decoy_probe.py showed that the needle task is solved by POSITION (nearest or first fact), not by key, in both
# the original and the distance-widened checkpoints. The experiment that matters for a key- or heading-addressed memory is
# training where the nearest fact is the right one only 1 time in n - mline/decoy_train.py - so it goes ahead of the
# replicate runs, which only tighten an error bar on the cliff's position.
#
#   D1  decoy fine-tune: 3 facts per example, the asked one random, total filler 0-128 split between them, 500 updates from the
#       original plain checkpoint; evaluated by the decoy probe (does the answer follow the KEY now?) and the distance sweeps.
#
# Waits for the GPU to be free (the control queue's running training keeps going on its own), one GPU job at a time.
# Idempotent: finished steps are skipped, an interrupted training resumes from latest.pt.
#
# Usage: nohup bash scripts/run_decoy_experiments.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."
PY="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
ROOT=results/M1-finetune-distance
M1P=E:/rlt-research/vm-mirror/results/M1/m1-plain-w1152-s117/selected.pt
GRID=0,32,64,72,80,96,128,160,192,224,256,320,384,448,512
DECOY_CONFIGS=0:0,0:10,10:0,0:30,30:0,15:15,50:0,0:50,25:25
LOG=$ROOT/queue.log
mkdir -p "$ROOT"

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }
wait_idle() { while powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/_gpu_busy.ps1 > /dev/null 2>&1; do sleep 30; done; }

decoy() {   # name init facts train_filler steps seed purpose
  local name=$1 init=$2 facts=$3 fill=$4 steps=$5 seed=$6 purpose=$7
  local out=$ROOT/$name
  mkdir -p "$out"
  if [ ! -f "$out/README.txt" ]; then
    cat > "$out/README.txt" <<EOF
EXPLORATORY FINE-TUNE, NOT A FROZEN CYCLE. One seed. Sizes an effect; is not evidence for or against a pre-registered claim.
$purpose
Recipe: mline.decoy_train from $init, $steps updates, batch 32 (8 needle rows = rate 0.25, packed into 512-byte rows), $facts
facts per example with the asked one chosen at random, total filler uniform on $fill bytes split between the facts, lr 3e-4,
fresh AdamW, fresh language-model data (seed $seed).
Read it with: decoy.json (does the answer follow the asked KEY?) against results/M1-diagnosis/decoy-plain.json and
ft384-plain-s217/decoy.json, and sweep-*.json (single-fact distance) against ft384-plain-s217/ and B6-*.
EOF
  fi
  if [ "$(lines "$out/probe.jsonl")" -lt "$steps" ]; then
    wait_idle
    log "BASLA egitim $name (yemli: $facts olgu, dolgu $fill, $steps adim)"
    $PY -B -m mline.decoy_train --out "$out" --init-from "$init" --facts "$facts" --train-filler "$fill" \
        --steps "$steps" --seed "$seed" >> "$out/train.log" 2>&1
  fi
  local n
  n=$(lines "$out/probe.jsonl")
  if [ "$n" -lt "$steps" ]; then log "HATA $name: egitim bitmedi ($n adim)"; return 1; fi
  if [ ! -f "$out/selected.pt" ]; then
    $PY -B -c "import torch; s=torch.load('$out/latest.pt', weights_only=True); torch.save(s['model'], '$out/selected.pt')" >> "$out/train.log" 2>&1
  fi
  if [ ! -f "$out/decoy.json" ]; then
    wait_idle
    $PY -B -m mline.decoy_probe --checkpoint "$out/selected.pt" --out "$out/decoy.json" --configs $DECOY_CONFIGS \
        --alphabets TRAIN_KEYS,EVAL_KEYS --items 128 --batch 32 --device cuda >> "$out/sweep.log" 2>&1
    log "BITTI $name: yem olgu probu"
  fi
  local keys tag
  for keys in TRAIN_KEYS EVAL_KEYS; do
    tag=$(echo $keys | tr 'A-Z_' 'a-z-')
    if [ ! -f "$out/sweep-$tag.json" ]; then
      wait_idle
      $PY -B -m mline.rope_probe --arm plain --checkpoint "$out/selected.pt" --out "$out/sweep-$tag.json" \
          --schemes identity:1 --fillers $GRID --alphabet $keys --items 128 --batch 32 >> "$out/sweep.log" 2>&1
    fi
  done
  log "BITTI $name"
}

log "TAKEOVER BASLADI"
wait_idle
decoy D1-decoy3-f128 "$M1P" 3 0,128 500 217 "D1: does decoy-aware training (3 facts, random asked one) teach matching on the KEY? The nearest fact is right only 1 time in 3."
log "TAKEOVER: yemli kosu bitti, kontrol kuyrugu devam ediyor"
bash scripts/run_distance_controls.sh
log "TAKEOVER BITTI"
