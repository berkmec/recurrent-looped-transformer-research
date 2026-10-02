#!/usr/bin/env bash
# E-L1P: does the L1 model use its feedback channel for language? Evaluation only, CPU, on two L1 snapshots (update 2,000 and the newest one at the time this runs).
# Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.8 (each snapshot's reading is printed by mline.feedback_probe).
# One CPU process, BelowNormal (the L1 training run owns the GPU and is not touched); idempotent (skips a snapshot whose result file exists).
#
# Usage: nohup bash scripts/run_el1p.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=results/L1-scratch-w928-s401
OUT=results/L1-feedback-probe
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
mkdir -p "$OUT"

NEWEST=$(ls "$RUN"/snapshots/step*.pt | sort | tail -n 1)
for snap in "$RUN/snapshots/step002000.pt" "$NEWEST"; do
  name=$(basename "$snap" .pt)
  if [ -f "$OUT/$name-feedback-probe.json" ] || [ ! -f "$snap" ]; then continue; fi
  log "BASLA E-L1P ($name; CPU, 1 parcacik): L1 geri besleme kanali kullanimi, egitimsiz"
  $PY -B -m mline.feedback_probe "$snap" --out "$OUT" > "$OUT/$name.txt" 2>&1
  if [ -f "$OUT/$name-feedback-probe.json" ]; then log "BITTI E-L1P $name (bkz $OUT/$name.txt)"; else log "HATA E-L1P $name: sonuc dosyasi yok (bkz $OUT/$name.txt)"; fi
done
