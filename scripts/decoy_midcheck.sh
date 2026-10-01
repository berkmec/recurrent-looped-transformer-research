#!/usr/bin/env bash
# Mid-training decoy probes for a decoy-aware fine-tune that runs on the GPU: CPU only, one thread, so they never compete with
# the trainer (scripts/_gpu_busy.ps1 ignores --device cpu). Whether key matching is emerging is then known at steps 250 / 500 /
# 750 instead of only when the run ends. The trainer overwrites latest.pt every 250 updates, so each step's checkpoint is
# copied as soon as it is written and checked to be exactly that step before it is probed.
#
# Usage: nohup bash scripts/decoy_midcheck.sh RUN STEPS [ALPHABETS] [KEY_LENGTHS] > /dev/null 2>&1 &
#   e.g.  bash scripts/decoy_midcheck.sh D3-mqar4-k14 250,500,750 TRAIN_KEYS 1,2,4
# Writes results/M1-finetune-distance/RUN/step<N>.pt and decoy-step<N>.json; progress goes to queue.log (BASLA/BITTI midcheck).
set -u
cd "$(dirname "$0")/.."
PY="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
RUN=$1
STEPS=$2
ALPHA=${3:-TRAIN_KEYS}
KEYLENS=${4:-4}
OUT=results/M1-finetune-distance/$RUN
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }
saved_step() { $PY -B -c "import sys,torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['step'])" "$1" 2>/dev/null; }

for step in ${STEPS//,/ }; do
  [ -f "$OUT/decoy-step$step.json" ] && continue
  while [ "$(lines "$OUT/probe.jsonl")" -lt "$step" ]; do sleep 30; done
  got=""
  for try in 1 2 3 4 5 6 7 8 9 10; do
    cp "$OUT/latest.pt" "$OUT/step$step.pt"
    got=$(saved_step "$OUT/step$step.pt")
    if [ "$got" = "$step" ]; then break; fi
    if [ -n "$got" ] && [ "$got" -gt "$step" ]; then break; fi      # latest.pt has moved past this step: cannot be recovered
    sleep 15                                                         # partial copy, or the save for this step is not written yet
  done
  if [ "$got" != "$step" ]; then
    log "HATA midcheck $RUN adim $step: kopyalanan kontrol noktasi adim '$got' (beklenen $step)"
    rm -f "$OUT/step$step.pt"
    continue
  fi
  log "BASLA midcheck $RUN adim $step (CPU, 1 parcacik; alfabe $ALPHA, anahtar uzunluklari $KEYLENS)"
  $PY -B -m mline.decoy_probe --checkpoint "$OUT/step$step.pt" --out "$OUT/decoy-step$step.json" --device cpu \
      --alphabets "$ALPHA" --key-lengths "$KEYLENS" --items 64 --batch 16 \
      --configs 0:0,10:0,30:0,0:30,15:15,50:0 >> "$OUT/midcheck.log" 2>&1
  log "BITTI midcheck $RUN adim $step"
done
