#!/usr/bin/env bash
# Relaunch the queues after scripts/pause_for_user.sh. Every runner is idempotent (finished runs are skipped) and every run resumes from its
# checkpoint; stage 1 resumes through scripts/run_erm_stage1_ut_resume.sh, which reads the G0.4b gate already measured instead of re-running it.
# Usage: bash scripts/resume_after_user.sh
set -u
cd "$(dirname "$0")/.."
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] [pause] $*" >> "$LOG"; }
log "SURDURULUYOR: Asama 1 (kayitli kapiyla), E-X1 bekleyicisi; E-RL0 ve E-BLIMP okumalari yoksa onlar da"
nohup bash scripts/run_erm_stage1_ut_resume.sh > /dev/null 2>&1 &
nohup bash scripts/run_x1.sh > /dev/null 2>&1 &
if ! grep -q "\[rl1\] E-RL1 kuyruk BITTI" "$LOG"; then nohup bash scripts/run_rl1.sh after-stage1 > /dev/null 2>&1 & fi
if [ ! -f results/RL0/reading.json ]; then nohup bash scripts/run_rl0.sh > /dev/null 2>&1 & fi
if [ ! -f results/BLIMP/reading.json ]; then nohup bash scripts/run_blimp.sh > /dev/null 2>&1 & fi
