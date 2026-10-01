#!/usr/bin/env bash
# Night resume for the D1 final (scheduled by Windows Task Scheduler at 00:00 local; independent of any Claude session).
# Resumes the resumable runner from its progress files and babysits it: if the GPU spills to host memory (power < 60 W and
# no progress write for > 5 min) the process is killed and relaunched from the progress file (<= 1 batch lost). Stops when
# final.json exists or after 9 hours. Everything is logged to d1-night-resume.log; the chain log gets the same stamps as before.
cd /e/rlt-research/2026-09-18-phase2 || exit 1
LOG=.superpowers/sdd/phase2/d1-night-resume.log
CHAIN=.superpowers/sdd/phase2/d1-final-chain.sh
PROG=results/D1-final/final-lm_schema-progress.jsonl
stamp(){ date -u +%FT%TZ; }
echo "$(stamp) night resume start (local $(date +%H:%M)); rows=$(wc -l < $PROG 2>/dev/null || echo 0)" >> "$LOG"
if [ -f results/D1-final/final.json ]; then echo "$(stamp) final.json already exists; nothing to do" >> "$LOG"; exit 0; fi
free_gb=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" | tr -d '\r')
echo "$(stamp) free RAM ${free_gb} GB" >> "$LOG"
launch(){ (nohup bash "$CHAIN" > /dev/null 2>&1 &); sleep 90; }
launch; restarts=0; start=$(date +%s)
while true; do
  [ -f results/D1-final/final.json ] && { echo "$(stamp) DONE: $(tail -1 .superpowers/sdd/phase2/d1-final-chain.log)" >> "$LOG"; break; }
  [ $(( $(date +%s) - start )) -gt 32400 ] && { echo "$(stamp) 9 h budget reached; leaving the job running unattended" >> "$LOG"; break; }
  pid=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -match 'python' -and \$_.CommandLine -match 'run_d1_final' -and \$_.WorkingSetSize -gt 50MB } | Select-Object -First 1).ProcessId" | tr -d '\r ')
  if [ -z "$pid" ]; then
    sleep 30; [ -f results/D1-final/final.json ] && continue
    if ! powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'run_d1_final' }).Count" | grep -qv '^0'; then
      echo "$(stamp) process gone without final.json; relaunch ($(tail -1 .superpowers/sdd/phase2/d1-final-chain.log))" >> "$LOG"; launch; restarts=$((restarts+1)); continue
    fi
  fi
  last=$(stat -c %Y "$PROG" 2>/dev/null || echo 0); for f in results/D1-final/*progress.jsonl; do m=$(stat -c %Y "$f"); [ "$m" -gt "$last" ] && last=$m; done
  stale=$(( $(date +%s) - last ))
  power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | tr -d ' ')
  if [ -n "$pid" ] && awk -v p="${power:-100}" -v s="$stale" 'BEGIN{exit !(p<60 && s>300)}'; then
    n=$(wc -l < "$PROG"); taskkill //F //PID "$pid" >/dev/null 2>&1; sleep 5
    echo "$(stamp) AUTO spill restart #$((restarts+1)) at $n rows (power ${power} W, stale ${stale}s)" >> "$LOG"
    echo "$(stamp) AUTO spill restart (night) at $n rows" >> .superpowers/sdd/phase2/d1-final-chain.log
    launch; restarts=$((restarts+1))
  fi
  sleep 120
done
echo "$(stamp) night resume end; restarts=$restarts" >> "$LOG"
