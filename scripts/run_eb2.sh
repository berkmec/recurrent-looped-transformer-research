#!/usr/bin/env bash
# E-B2: finite-time growth of a single relative kick to the carried state (float64) for the saved S4 / S5 arms: does the kick grow, shrink or die out,
# and does it flip predictions? Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.4. Evaluation only (no training), dev split.
# One CPU process, BelowNormal; idempotent (skips when the result file exists).
#
# Usage: nohup bash scripts/run_eb2.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ -f "$OUT/state-growth.json" ]; then exit 0; fi
log "BASLA E-B2 (CPU, 1 parcacik): durum tekmesinin buyumesi, 6 kayitli model, egitimsiz"
$PY -B -m horizon.state_growth \
  "$OUT/perm4-core-w64-s1-cur.pt" "$OUT/perm5-core-w128-s1-cur.pt" \
  "$OUT/perm4-gru-w64-s1-cur.pt" "$OUT/perm4-lstm-w64-s1-cur.pt" "$OUT/perm5-gru-w256-s1-cur.pt" "$OUT/perm5-lstm-w208-s1-cur.pt" \
  --out "$OUT/state-growth.json" > "$OUT/state-growth.txt" 2>&1
if [ -f "$OUT/state-growth.json" ]; then log "BITTI E-B2 (bkz $OUT/state-growth.txt)"; else log "HATA E-B2: sonuc dosyasi yok (bkz $OUT/state-growth.txt)"; fi
