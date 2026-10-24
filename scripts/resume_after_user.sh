#!/usr/bin/env bash
# Relaunch the queues after scripts/pause_for_user.sh. User rule (2026-10-24 ~10:10, note 5.70): no model is trained on the CNN corpus
# again (mline-data/cnn-bytes); datasets are chosen by purpose. Stage 1's old runs, E-X1 and E-RL1 all trained on that corpus, so they are
# not relaunched here. What comes back: E-RM0R8 (synthetic toy data, note 5.72), E-RM1W (FineWeb-Edu, note 5.73) and E-SC1 (FineWeb-Edu,
# note 5.74; waits for E-RM1W), the last two resuming from their checkpoints, each only if its queue has not finished.
# Usage: bash scripts/resume_after_user.sh
set -u
cd "$(dirname "$0")/.."
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] [pause] $*" >> "$LOG"; }
log "SURDURULUYOR: E-RM0R8 (yapay veri) ve E-RM1W (FineWeb-Edu); CNN korpusuyla egitim kullanici kuraliyla kapali"
if [ -f scripts/run_erm_r8.sh ] && ! grep -q "\[erm8\] E-RM0R8 kuyruk BITTI" "$LOG"; then nohup bash scripts/run_erm_r8.sh > /dev/null 2>&1 & fi
if [ -f scripts/run_erm1w.sh ] && ! grep -q "\[erm1w\] E-RM1W kuyrugu BITTI" "$LOG"; then nohup bash scripts/run_erm1w.sh > /dev/null 2>&1 & fi
if [ -f scripts/run_esc1.sh ] && ! grep -q "\[esc1\] E-SC1 kuyrugu BITTI" "$LOG"; then nohup bash scripts/run_esc1.sh > /dev/null 2>&1 & fi
