#!/usr/bin/env bash
# L1: the long run - a from-scratch ~20M-parameter model (width 928, 8 heads, plain arm) on 500M byte-tokens, NOT stopped for a plateau.
#
# Why it is not stopped: phase transitions in this project arrive late and sharply (a small Core sat at chance for 2000 updates and was at
# 0.96 by 2500; the 30M fine-tunes moved between 500 and 1000), so a run judged at checkpoints would be stopped where it was about to change.
# It ends when its 500M tokens are spent, or on a repeating non-finite loss (DIVERGED.txt), or when a person stops it.
#
# This script only keeps it alive: it waits for the GPU, starts mline.longrun (which resumes from its own latest.pt), and starts it again
# if the process dies without finishing. A power cut also kills this script - a Windows scheduled task that starts it again at logon is
# what would survive one (it needs the owner's approval; nothing here creates it).
#
# Usage: nohup bash scripts/run_longrun.sh OUT_DIR [RECIPE] > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
OUT=${1:?output directory}
RECIPE=${2:-m2-draft}
PY="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
wait_idle() { while powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/_gpu_busy.ps1 > /dev/null 2>&1; do sleep 30; done; }
mkdir -p "$OUT"
if [ ! -f "$OUT/README.txt" ]; then
  cat > "$OUT/README.txt" <<'README'
L1 - THE LONG RUN. Exploratory, one seed, plain arm. Written before any update ran.

Question: does a from-scratch ~20.3M-parameter RLT (width 928, 8 heads), trained on 500M byte-tokens (about 2.4 epochs of cnn_dailymail) with
25 % synthetic rows (single-fact, MQAR with questions drawn with replacement over random and natural keys, distance), learn
  (a) key-addressed retrieval that holds on letters it never keyed on and on real words and names,
  (b) retrieval across at least 384 bytes,
while it keeps learning language?

It is NOT stopped for a plateau. A flat loss is not a reason to stop: transitions here arrive late and sharply.

Logged every update (probe.jsonl): loss, loss_lm, loss_synthetic, answer_loss (loss on the answer digits only), first_digit_accuracy, lr, grad norm.
Every 1000 updates (the 'eval' field): validation bits per byte; train_k4_first / exact (training letters), held_k4_first / exact (T-Z, never keys),
natural_first / exact (real validation words and names that are not in the training key list); single_fact_exact@{0,64,128,256,384,448};
attention_conditionality_mean / best. A model snapshot (snapshots/stepNNNNNN.pt) every 1000 updates.

Pre-registered thresholds (chance for a first digit is 0.1):
  key matching LEARNED  : train_k4_first >= 0.8
  GENERALISES           : held_k4_first >= 0.6 AND natural_first >= 0.6, at the same eval
  REACH                 : single_fact_exact@384 >= 0.9
  language              : bits per byte is reported, with no threshold - language quality is not the claim.
A TRANSITION is dated by the first update at which the 500-update moving average of first_digit_accuracy exceeds 0.5.
The final report gives every curve and, for each threshold, the update at which it was first crossed or 'never' - not only the last value.
README
fi
tries=0
while true; do
  if [ -f "$OUT/complete.json" ]; then log "BITTI L1 $OUT (500M token tamam)"; break; fi
  if [ -f "$OUT/DIVERGED.txt" ]; then log "HATA L1 $OUT: DIVERGED.txt var, bkz $OUT/DIVERGED.txt"; break; fi
  tries=$((tries + 1))
  if [ "$tries" -gt 30 ]; then log "HATA L1 $OUT: 30 yeniden baslatmadan sonra vazgecildi"; break; fi
  wait_idle
  log "BASLA L1 $OUT (tarif $RECIPE, deneme $tries)"
  $PY -B -m mline.longrun --out "$OUT" --recipe "$RECIPE" --tokens 500000000 >> "$OUT/train.log" 2>&1
  log "L1 $OUT: surec cikti (kod $?), $(wc -l < "$OUT/probe.jsonl" 2>/dev/null || echo 0) adim"
  sleep 30
done
