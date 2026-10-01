#!/usr/bin/env bash
# Waits for the R6 driver to finish (ALL DONE or FAILED), then launches the R5 selection worker on the GPU.
cd /e/rlt-research/2026-09-17-diagnostics || exit 1
PY="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
LOG=.superpowers/sdd/review-diagnostics-plan/r6-driver.log
OUT='E:\rlt-research\2026-09-17-diagnostics\results\training'
until grep -qE "ALL DONE|FAILED" "$LOG"; do sleep 10; done
sleep 5
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
nohup "$PY" -B -m review_cycle.campaign --out "$OUT" --role selection > results/training/processes/selection.stdout.log 2> results/training/processes/selection.stderr.log &
PID=$!
cat > results/training/processes/launch-02.json <<JSON
[
  {
    "name": "selection",
    "pid": $PID,
    "started_utc": "$STARTED",
    "r6_driver_final_line": "$(grep -E 'ALL DONE|FAILED' "$LOG" | tail -1)",
    "arguments": ["-B", "-m", "review_cycle.campaign", "--out", "E:\\rlt-research\\2026-09-17-diagnostics\\results\\training", "--role", "selection"]
  }
]
JSON
echo "$STARTED launched selection pid=$PID" >> results/training/processes/launch-02.log
wait $PID; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) selection exited code=$?" >> results/training/processes/launch-02.log
