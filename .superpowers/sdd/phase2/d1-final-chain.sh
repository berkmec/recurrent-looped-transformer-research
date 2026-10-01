#!/usr/bin/env bash
# Step 4 of the sequential restart: D1 final on the resumable runner. Exit code captured before any command substitution.
cd /e/rlt-research/2026-09-18-phase2 || exit 1
V="E:/rlt-research/.venv-phase2/Scripts/python.exe"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=.superpowers/sdd/phase2/d1-final-chain.log
stamp(){ date -u +%FT%TZ; }
echo "$(stamp) D1 final start" >> "$LOG"
"$V" -u -B -m language.run_d1_final --final --out results/D1-final > .superpowers/sdd/phase2/d1-final.log 2>&1; rc=$?
echo "$(stamp) D1 final rc=$rc final.json=$([ -f results/D1-final/final.json ] && echo yes || echo no) receipt=$(grep -o '"status": "[A-Z]*"' results/D1-final/receipt.json 2>/dev/null | head -1)" >> "$LOG"
