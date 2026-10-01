#!/usr/bin/env bash
# B1 rerun v2 (replaces b1-rerun-chain.sh, which had only been waiting): after the D1 chain finishes, freeze B1 with the
# development-selected overrides, run the contract tests, run the final, and write an E12 receipt (run id, exit code,
# item count, test result, artifact hashes). Exit codes are captured BEFORE any command substitution.
cd /e/rlt-research/2026-09-18-phase2 || exit 1
DAX="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
LOG=.superpowers/sdd/phase2/b1-rerun-chain.log
RUN_ID="b1-final-rerun-$(date -u +%Y%m%dT%H%M%SZ)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp(){ date -u +%FT%TZ; }
echo "$(stamp) v2 armed run_id=$RUN_ID; waiting for 'D1 final exit'" >> "$LOG"
until grep -q "D1 final exit" .superpowers/sdd/phase2/d1-chain.log 2>/dev/null; do sleep 30; done
echo "$(stamp) D1 chain finished; B1 freeze (patched reranker)" >> "$LOG"
OVR='{"dense_max_length": 256, "dense_instruction": "Given a web search query, retrieve relevant passages that answer the query", "rerank_depth": 50}'
"$DAX" -B -m semantic_memory.retrieval --freeze --out results/B1 --protocol-json "$OVR" >> "$LOG" 2>&1; rc=$?
echo "$(stamp) freeze rc=$rc" >> "$LOG"; [ $rc -ne 0 ] && exit 2
"$DAX" -B -m unittest tests.test_retrieval_contract > .superpowers/sdd/phase2/b1-rerun-tests.log 2>&1; trc=$?
echo "$(stamp) contract tests rc=$trc; B1 final" >> "$LOG"
"$DAX" -u -B -m semantic_memory.retrieval --final --out results/B1 > .superpowers/sdd/phase2/b1-final.log 2>&1; rc=$?
n=$("$DAX" -c "import json; print(len(json.load(open('results/B1/final.json', encoding='utf8'))['arms']))" 2>/dev/null); [ -z "$n" ] && n=0
tp=false; [ "$trc" -eq 0 ] && tp=true
"$DAX" -B -m audit.receipt --out results/B1/receipt.json --run-id "$RUN_ID" --returncode "$rc" --expected-items 5 --completed-items "$n" --tests-passed "$tp" \
  --artifact final=results/B1/final.json --artifact frozen=results/B1/frozen.json >> "$LOG" 2>&1
echo "$(stamp) B1 final rc=$rc arms=$n tests_rc=$trc receipt=$(grep -o '"status": "[A-Z]*"' results/B1/receipt.json 2>/dev/null | head -1)" >> "$LOG"
