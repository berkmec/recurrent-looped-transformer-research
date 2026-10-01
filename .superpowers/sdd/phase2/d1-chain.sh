#!/usr/bin/env bash
cd /e/rlt-research/2026-09-18-phase2 || exit 1
V="E:/rlt-research/.venv-phase2/Scripts/python.exe"
LOG=.superpowers/sdd/phase2/d1-chain.log
until grep -q "B1 final exit" .superpowers/sdd/phase2/gpu-chain.log 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) B1 final finished; D1 freeze" >> "$LOG"
"$V" -B -m language.run_d1 --freeze --out results/D1 >> "$LOG" 2>&1 || { echo "D1 FREEZE FAILED" >> "$LOG"; exit 2; }
echo "$(date -u +%FT%TZ) D1 development (32, 200)" >> "$LOG"
"$V" -u -B -m language.run_d1 --development --out results/D1 > .superpowers/sdd/phase2/d1-development.log 2>&1
echo "$(date -u +%FT%TZ) development exit=$? ; D1 final" >> "$LOG"
"$V" -u -B -m language.run_d1 --final --out results/D1 > .superpowers/sdd/phase2/d1-final.log 2>&1
echo "$(date -u +%FT%TZ) D1 final exit=$?" >> "$LOG"
