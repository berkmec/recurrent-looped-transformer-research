#!/usr/bin/env bash
# E-Q: are the first errors of the trained S4 / S5 arms DETERMINISTIC (the same few (state, token) transitions wrong again and again) or NOISE-LIKE? Evaluation only.
# Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.9. One CPU process, BelowNormal; idempotent (skips existing results).
#
# Usage: nohup bash scripts/run_eq.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ ! -f "$OUT/pair-errors-recurrent.json" ]; then
  log "BASLA E-Q (CPU, 1 parcacik): hata cifti dagilimi, GRU/LSTM S4+S5, egitimsiz"
  $PY -B -m horizon.pair_errors "$OUT/perm4-gru-w64-s1-cur.pt" "$OUT/perm4-lstm-w64-s1-cur.pt" "$OUT/perm4-gru-w128-s1-cur.pt" "$OUT/perm4-lstm-w112-s1-cur.pt" \
    "$OUT/perm5-gru-w256-s1-cur.pt" "$OUT/perm5-lstm-w208-s1-cur.pt" --n 3000 --length 64 --windows 1-64 --out "$OUT/pair-errors-recurrent.json" > "$OUT/pair-errors-recurrent.txt" 2>&1
  if [ -f "$OUT/pair-errors-recurrent.json" ]; then log "BITTI E-Q recurrent (bkz $OUT/pair-errors-recurrent.txt)"; else log "HATA E-Q recurrent: sonuc dosyasi yok"; fi
fi
if [ ! -f "$OUT/pair-errors-core.json" ]; then
  log "BASLA E-Q (CPU, 1 parcacik): hata cifti dagilimi, S5 Core, egitimsiz"
  $PY -B -m horizon.pair_errors "$OUT/perm5-core-w128-s1-cur.pt" --n 1500 --length 256 --windows 1-64,128-256 --out "$OUT/pair-errors-core.json" > "$OUT/pair-errors-core.txt" 2>&1
  if [ -f "$OUT/pair-errors-core.json" ]; then log "BITTI E-Q core (bkz $OUT/pair-errors-core.txt)"; else log "HATA E-Q core: sonuc dosyasi yok"; fi
fi
