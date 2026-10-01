#!/usr/bin/env bash
# Second lr level for U2c: the same 18 conditions at lr 0.0003. Starts only when the lr probe frees its slot,
# so at most two CPU jobs run at once (the probe, then this, alongside U2c-v2).
cd /e/rlt-research/2026-09-18-phase2 || exit 1
PY="C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe"
LOG=.superpowers/sdd/phase2/u2c-lr0003.log
stamp(){ date -u +%FT%TZ; }
echo "$(stamp) waiting for the lr probe to finish before starting the 0.0003 level" >> "$LOG"
while true; do
  n=$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -match 'python' -and \$_.CommandLine -match 'lr_probe' -and \$_.WorkingSetSize -gt 50MB }).Count" | tr -d '\r ')
  [ "${n:-0}" -eq 0 ] && break
  sleep 120
done
free=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" | tr -d '\r')
echo "$(stamp) probe finished; free RAM ${free} GB; starting 18 conditions at lr 0.0003" >> "$LOG"
"$PY" -u -B -m horizon.gated_sweep --worker 0 --slots 1 --out results/U2c-lr0003 >> "$LOG" 2>&1
echo "$(stamp) lr0003 worker exited rc=$?" >> "$LOG"
