#!/usr/bin/env bash
# Relaunch the queues after scripts/pause_for_user.sh. User rule (2026-10-24 ~10:10): no model is trained on the CNN corpus again
# (mline-data/cnn-bytes); datasets are chosen by purpose. Stage 1's remaining runs, E-X1 and E-RL1 all train on that corpus, so they are
# not relaunched here; they wait for the natural-language corpus (note 5.71). Only runs that train on purpose-built synthetic data come back.
# Usage: bash scripts/resume_after_user.sh
set -u
cd "$(dirname "$0")/.."
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] [pause] $*" >> "$LOG"; }
log "SURDURULUYOR: yalniz yapay veriyle egitilen kuyruklar (CNN korpusu kullanici kuraliyla kapali; Asama 1, E-X1, E-RL1 yeni korpusu bekler)"
if [ -f scripts/run_erm_r8.sh ] && ! grep -q "\[erm8\] E-RM0R8 kuyruk BITTI" "$LOG"; then nohup bash scripts/run_erm_r8.sh > /dev/null 2>&1 & fi
