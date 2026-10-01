#!/usr/bin/env bash
# Waits for the R5 selection worker to exit with code 0, then launches the final heldout evaluation on the frozen device.
cd /e/rlt-research/2026-09-17-diagnostics || exit 1
PY="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
LOG=results/training/processes/launch-02.log
OUT='E:\rlt-research\2026-09-17-diagnostics\results\training'
until grep -q "selection exited" "$LOG" 2>/dev/null; do sleep 15; done
if ! grep -q "selection exited code=0" "$LOG"; then echo "$(date -u +%FT%TZ) selection did not exit 0; evaluation NOT launched" >> results/training/processes/launch-03.log; exit 3; fi
sleep 5
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
nohup "$PY" -B -m review_cycle.campaign --out "$OUT" --role evaluate > results/training/processes/evaluate.stdout.log 2> results/training/processes/evaluate.stderr.log &
PID=$!
cat > results/training/processes/launch-03.json <<JSON
[
  {
    "name": "evaluate",
    "pid": $PID,
    "started_utc": "$STARTED",
    "gate": "launched only after launch-02.log recorded selection exited code=0",
    "arguments": ["-B", "-m", "review_cycle.campaign", "--out", "E:\\rlt-research\\2026-09-17-diagnostics\\results\\training", "--role", "evaluate"]
  }
]
JSON
echo "$STARTED launched evaluate pid=$PID" >> results/training/processes/launch-03.log
wait $PID; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) evaluate exited code=$?" >> results/training/processes/launch-03.log
