#!/usr/bin/env bash
# Local queue for the distance experiments that need NO new code (report: "Yinelemeli modellerde uzak bilgi tasima").
#
#   B2  control      the fine-tune recipe with the distance range UNCHANGED (filler 0-64): does the cliff move without
#                    widening the range? Without this arm the claim "widening the range moved the cliff" is unproven,
#                    because needle share, learning rate, optimizer reset and fresh data all changed together.
#   B6  dose-response  trained maximum 192 and 128: the cliff should land just beyond each maximum.
#   B5  replicates   the main recipe (0-384) with two new data seeds, and from the hard_st checkpoint (the same function
#                    as plain, because its gate never closed), with the gate checked once more on the result.
#
# One GPU job at a time. Idempotent: a finished step is skipped, an interrupted training resumes from latest.pt.
# Each run is a DEVELOPMENT probe (mline.emergence): one seed, no frozen protocol, evidence about sizes of effects only.
#
# Usage: nohup bash scripts/run_distance_controls.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."
PY="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
ROOT=results/M1-finetune-distance
M1P=E:/rlt-research/vm-mirror/results/M1/m1-plain-w1152-s117/selected.pt
M1H=E:/rlt-research/vm-mirror/results/M1/m1-hard_st-w1152-s117/selected.pt
GRID=0,32,64,72,80,96,128,160,192,224,256,320,384,448,512
LOG=$ROOT/queue.log
mkdir -p "$ROOT"

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }

run() {   # name arm init train_filler seed purpose
  local name=$1 arm=$2 init=$3 fill=$4 seed=$5 purpose=$6
  local out=$ROOT/$name
  mkdir -p "$out"
  if [ ! -f "$out/README.txt" ]; then
    cat > "$out/README.txt" <<EOF
EXPLORATORY FINE-TUNE, NOT A FROZEN CYCLE. One seed. Sizes an effect; is not evidence for or against a pre-registered claim.
$purpose
Recipe: mline.emergence from $init, 500 updates, batch 32 (8 needle rows = rate 0.25, packed into 512-byte rows), needle
filler uniform on $fill bytes, lr 3e-4, fresh AdamW, fresh language-model data (seed $seed), arm $arm.
Read it with: sweep-train-keys.json and sweep-eval-keys.json, against ft384-plain-s217/ (the original 0-384 run) and
results/M1-diagnosis/distance-sweep-plain.json (the checkpoint it started from).
EOF
  fi
  if [ "$(lines "$out/probe.jsonl")" -lt 500 ]; then
    log "BASLA egitim $name ($fill, tohum $seed)"
    $PY -B -m mline.emergence --out "$out" --arm "$arm" --init-from "$init" --steps 500 --batch 32 --every 250 \
        --seed "$seed" --needle-rate 0.25 --train-filler "$fill" --lr 3e-4 >> "$out/train.log" 2>&1
  fi
  local n
  n=$(lines "$out/probe.jsonl")
  if [ "$n" -lt 500 ]; then log "HATA $name: egitim bitmedi ($n adim)"; return 1; fi
  if [ ! -f "$out/selected.pt" ]; then
    $PY -B -c "import torch; s=torch.load('$out/latest.pt', weights_only=True); torch.save(s['model'], '$out/selected.pt')" >> "$out/train.log" 2>&1
  fi
  local keys tag
  for keys in TRAIN_KEYS EVAL_KEYS; do
    tag=$(echo $keys | tr 'A-Z_' 'a-z-')
    if [ ! -f "$out/sweep-$tag.json" ]; then
      $PY -B -m mline.rope_probe --arm "$arm" --checkpoint "$out/selected.pt" --out "$out/sweep-$tag.json" \
          --schemes identity:1 --fillers $GRID --alphabet $keys --items 128 --batch 32 >> "$out/sweep.log" 2>&1
    fi
  done
  if [ ! -f "$out/sweep-eval-keys.json" ]; then log "HATA $name: supurme yok"; return 1; fi
  log "BITTI $name"
}

log "KUYRUK BASLADI"
run B2-control-s217      plain   "$M1P" 0,64  217 "B2 CONTROL: same recipe as ft384-plain-s217 but the distance range is NOT widened (0-64)."
run B6-ft192-s217        plain   "$M1P" 0,192 217 "B6 dose-response: trained maximum 192 (the main run used 384)."
run B6-ft128-s217        plain   "$M1P" 0,128 217 "B6 dose-response: trained maximum 128."
run B5-ft384-s318        plain   "$M1P" 0,384 318 "B5 replicate of the main recipe (0-384) with a new data seed."
run B5-ft384-s419        plain   "$M1P" 0,384 419 "B5 replicate of the main recipe (0-384) with a second new data seed."
run B5-ft384-hardst-s217 hard_st "$M1H" 0,384 217 "B5 replicate from the hard_st checkpoint: the same function as plain (its gate never closed), independently trained."

if [ -f "$ROOT/B5-ft384-hardst-s217/selected.pt" ] && [ ! -f "$ROOT/B5-ft384-hardst-s217/channel-hard_st.json" ]; then
  $PY -B -m mline.channel_probe --arm hard_st --checkpoint "$ROOT/B5-ft384-hardst-s217/selected.pt" \
      --out "$ROOT/B5-ft384-hardst-s217/channel-hard_st.json" --fillers 64,128,384 --modes full,gate_open >> "$LOG" 2>&1
  log "BITTI kapi kontrolu (hard_st tekrari)"
fi
log "KUYRUK BITTI"
