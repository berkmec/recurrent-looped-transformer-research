#!/usr/bin/env bash
# B2 multi-hop retrieval: development (100 train queries) then final (all dev queries); E12 receipt. Exit codes captured first.
cd /e/rlt-research/2026-09-18-phase2 || exit 1
DAX="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
LOG=.superpowers/sdd/phase2/b2-chain.log
RUN_ID="b2-$(date -u +%Y%m%dT%H%M%SZ)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp(){ date -u +%FT%TZ; }
echo "$(stamp) B2 start run_id=$RUN_ID" >> "$LOG"
"$DAX" -B -m unittest tests.test_multihop tests.test_run_b2 > .superpowers/sdd/phase2/b2-tests.log 2>&1; trc=$?
echo "$(stamp) contract tests rc=$trc; development" >> "$LOG"
"$DAX" -u -B -m semantic_memory.run_b2 --development --out results/B2 > .superpowers/sdd/phase2/b2-development.log 2>&1; rc=$?
echo "$(stamp) development rc=$rc development.json=$([ -f results/B2/development.json ] && echo yes || echo no); final" >> "$LOG"
"$DAX" -u -B -m semantic_memory.run_b2 --final --out results/B2 > .superpowers/sdd/phase2/b2-final.log 2>&1; rc=$?
tp=false; [ "$trc" -eq 0 ] && tp=true
n=0; [ -f results/B2/development.json ] && n=$((n+1)); [ -f results/B2/final.json ] && n=$((n+1))
"$DAX" -B -m audit.receipt --out results/B2/receipt.json --run-id "$RUN_ID" --returncode "$rc" --expected-items 2 --completed-items "$n" --tests-passed "$tp" \
  --artifact final=results/B2/final.json --artifact development=results/B2/development.json --artifact frozen=results/B2/frozen.json >> "$LOG" 2>&1
echo "$(stamp) B2 final rc=$rc items=$n tests_rc=$trc receipt=$(grep -o '"status": "[A-Z]*"' results/B2/receipt.json 2>/dev/null | head -1)" >> "$LOG"
