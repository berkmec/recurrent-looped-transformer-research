#!/usr/bin/env bash
# E-L1K: how long does ONE wrong feedback state hurt the L1 model (a kick at a single decoder step, per-position paired difference against the untouched run)? Evaluation only,
# CPU, on explicit L1 snapshots (default: the 3,000-update one that E-L1P read). Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.19
# (each snapshot's reading is printed by mline.feedback_kick). One CPU process, BelowNormal (the L1 training run owns the GPU and is not touched); idempotent.
#
# Usage: nohup bash scripts/run_el1k.sh [snapshot ...] > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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

if [ "$#" -gt 0 ]; then SNAPS=("$@"); else SNAPS=("$RUN/snapshots/step003000.pt"); fi
for snap in "${SNAPS[@]}"; do
  name=$(basename "$snap" .pt)
  if [ -f "$OUT/$name-kick.json" ] || [ ! -f "$snap" ]; then continue; fi
  log "BASLA E-L1K ($name; CPU, 1 parcacik): tek adimlik yanlis geri besleme durumunun hasari ne kadar suruyor, egitimsiz"
  $PY -B -m mline.feedback_kick "$snap" --out "$OUT" > "$OUT/$name-kick.txt" 2>&1
  if [ -f "$OUT/$name-kick.json" ]; then log "BITTI E-L1K $name (bkz $OUT/$name-kick.txt)"; else log "HATA E-L1K $name: sonuc dosyasi yok (bkz $OUT/$name-kick.txt)"; fi
done
bash scripts/run_ep2x.sh      # E-P2 descriptive extras on the annealed S5 Core (note 5.11); chained here, before this script has started: it runs in the slot this chain already owns
