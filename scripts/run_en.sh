#!/usr/bin/env bash
# E-N: neutral runs (identity padding) for the saved S4 / S5 arms: is the per-step first-error rate a matter of TIME (drift while idle) or of WORK (every real update)?
# Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.7 (each model's reading is printed by horizon.neutral_runs).
# Evaluation only (no training), dev split. One CPU process, BelowNormal; idempotent (skips when the result file exists).
#
# Usage: nohup bash scripts/run_en.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ -f "$OUT/neutral-runs.json" ]; then exit 0; fi
log "BASLA E-N (CPU, 1 parcacik): kimlik-dolgu (notr kosu), 6 kayitli model, egitimsiz"
$PY -B -m horizon.neutral_runs \
  "$OUT/perm5-core-w128-s1-cur.pt" "$OUT/perm4-core-w64-s1-cur.pt" \
  "$OUT/perm4-gru-w64-s1-cur.pt" "$OUT/perm4-lstm-w64-s1-cur.pt" "$OUT/perm5-gru-w256-s1-cur.pt" "$OUT/perm5-lstm-w208-s1-cur.pt" \
  --out "$OUT/neutral-runs.json" > "$OUT/neutral-runs.txt" 2>&1
if [ -f "$OUT/neutral-runs.json" ]; then log "BITTI E-N (bkz $OUT/neutral-runs.txt)"; else log "HATA E-N: sonuc dosyasi yok (bkz $OUT/neutral-runs.txt)"; fi
