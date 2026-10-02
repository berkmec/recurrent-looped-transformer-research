#!/usr/bin/env bash
# E-B: state-noise margin curves of the saved S4 / S5 arms (Core, GRU, LSTM): first-error hazard versus relative noise sigma on the carried state.
# Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.1. Evaluation only (no training), dev split, tolerates nothing but CPU.
# One CPU process, BelowNormal; idempotent (skips when the result file exists).
#
# Usage: nohup bash scripts/run_eb.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ -f "$OUT/state-noise-margin.json" ]; then exit 0; fi
log "BASLA E-B (CPU, 1 parcacik): durum gurultusu marj egrileri, 8 kayitli model, egitimsiz"
$PY -B -m horizon.state_noise \
  "$OUT/perm4-core-w64-s1-cur.pt" "$OUT/perm5-core-w128-s1-cur.pt" \
  "$OUT/perm4-gru-w64-s1-cur.pt" "$OUT/perm4-gru-w128-s1-cur.pt" "$OUT/perm4-lstm-w64-s1-cur.pt" "$OUT/perm4-lstm-w112-s1-cur.pt" \
  "$OUT/perm5-gru-w256-s1-cur.pt" "$OUT/perm5-lstm-w208-s1-cur.pt" \
  --out "$OUT/state-noise-margin.json" > "$OUT/state-noise-margin.txt" 2>&1
if [ -f "$OUT/state-noise-margin.json" ]; then log "BITTI E-B (bkz $OUT/state-noise-margin.txt)"; else log "HATA E-B: sonuc dosyasi yok (bkz $OUT/state-noise-margin.txt)"; fi
