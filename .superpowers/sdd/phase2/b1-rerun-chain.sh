#!/usr/bin/env bash
# Rerun B1 final after the D1 chain finishes; exit codes captured BEFORE any command substitution.
cd /e/rlt-research/2026-09-18-phase2 || exit 1
DAX="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
LOG=.superpowers/sdd/phase2/b1-rerun-chain.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "D1 final exit" .superpowers/sdd/phase2/d1-chain.log 2>/dev/null; do sleep 30; done
stamp(){ date -u +%FT%TZ; }
echo "$(stamp) D1 chain finished; B1 freeze (patched reranker)" >> "$LOG"
OVR='{"dense_max_length": 256, "dense_instruction": "Given a web search query, retrieve relevant passages that answer the query", "rerank_depth": 50}'
"$DAX" -B -m semantic_memory.retrieval --freeze --out results/B1 --protocol-json "$OVR" >> "$LOG" 2>&1; rc=$?
echo "$(stamp) freeze rc=$rc" >> "$LOG"; [ $rc -ne 0 ] && exit 2
"$DAX" -u -B -m semantic_memory.retrieval --final --out results/B1 > .superpowers/sdd/phase2/b1-final.log 2>&1; rc=$?
echo "$(stamp) B1 final rc=$rc final.json=$([ -f results/B1/final.json ] && echo yes || echo no)" >> "$LOG"
