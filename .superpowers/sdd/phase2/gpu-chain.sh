#!/usr/bin/env bash
cd /e/rlt-research/2026-09-18-phase2 || exit 1
DAX="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
V="E:/rlt-research/.venv-phase2/Scripts/python.exe"
LOG=.superpowers/sdd/phase2/gpu-chain.log
until [ -f results/B1-development/development.json ]; do sleep 30; done
echo "$(date -u +%FT%TZ) B1 development done; starting 4B calibration" >> "$LOG"
"$V" -u protocols/lm_calibration.py > .superpowers/sdd/phase2/lm-calibration.log 2>&1
echo "$(date -u +%FT%TZ) calibration exit=$? ; freezing B1 with the development selection" >> "$LOG"
OVR=$("$DAX" -c "import json;s=json.load(open('results/B1-development/development.json'))['selected'];p=json.load(open('results/B1-development/development.json'));print(json.dumps(dict(dense_max_length=s['max_length'],dense_instruction=('Given a web search query, retrieve relevant passages that answer the query' if s['instruction'] else None),rerank_depth=s['rerank_depth'])))")
echo "$(date -u +%FT%TZ) overrides $OVR" >> "$LOG"
"$DAX" -B -m semantic_memory.retrieval --freeze --out results/B1 --protocol-json "$OVR" >> "$LOG" 2>&1 || { echo "FREEZE FAILED" >> "$LOG"; exit 2; }
echo "$(date -u +%FT%TZ) B1 final starting" >> "$LOG"
"$DAX" -u -B -m semantic_memory.retrieval --final --out results/B1 > .superpowers/sdd/phase2/b1-final.log 2>&1
echo "$(date -u +%FT%TZ) B1 final exit=$?" >> "$LOG"
