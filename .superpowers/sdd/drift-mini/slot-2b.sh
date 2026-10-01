#!/usr/bin/env bash
cd /e/rlt-research/2026-09-18-drift || exit 1
PY="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
LOG=results/drift/workers/slot-2b.jsonl
ev(){ printf '{"state": "%s", "condition": "%s", "utc": %s, "note": "%s"}\n' "$1" "$2" "$(date -u +%s)" "$3" >> "$LOG"; }
ev STARTED "-" "replacement for crashed slot 2; runs remaining slot-2 queue via frozen CLI"
for cid in drift-parity-idinv-s117 drift-parity-win16-s143 drift-parity-base-s191 drift-parity-reread-s191; do
  ev CONDITION_STARTED "$cid" ""
  if "$PY" -B -m drift.run --run-one "$cid" --out results/drift > "results/drift/processes/2b-$cid.stdout.log" 2> "results/drift/processes/2b-$cid.stderr.log"; then
    if "$PY" -B -m drift.run --evaluate-one "$cid" --out results/drift > "results/drift/processes/2b-$cid.eval.stdout.log" 2> "results/drift/processes/2b-$cid.eval.stderr.log"; then
      ev CONDITION_COMPLETE "$cid" ""
    else ev FAILED "$cid" "evaluate failed; see processes log"; fi
  else ev FAILED "$cid" "training failed; evidence preserved in condition directory"; fi
done
ev COMPLETE "-" ""
