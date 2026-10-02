#!/usr/bin/env bash
# E-X: how far does exactness extend past the trained length (16)? Evaluation only: saved S4 arms on 2,048 ... 32,768 fresh dev positions (up to 2,048x the trained length).
# Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.13 (horizon/long_eval.py::reading: CLEAN / FLOOR / BREAK / SPARSE / SHORT).
# One CPU process, BelowNormal, idempotent (skips a finished group). The Core is evaluated on fewer sequences (n = 30, batch 10): its decoder loop is sequential.
#
# Usage: nohup bash scripts/run_ex.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ ! -f "$OUT/long-eval-recurrent.json" ]; then
  log "BASLA E-X (CPU, 1 parcacik): uzun uzunluk degerlendirmesi, GRU/LSTM S4 (sabit lr, lr azaltmali, w128/w112), 2048-32768 konum"
  $PY -B -m horizon.long_eval "$OUT/perm4-gru-w64-s1-cur.pt" "$OUT/perm4-lstm-w64-s1-cur.pt" "$OUT/perm4-gru-w64-s1-cur-lrdecay0.3.pt" "$OUT/perm4-lstm-w64-s1-cur-lrdecay0.3.pt" \
    "$OUT/perm4-gru-w128-s1-cur.pt" "$OUT/perm4-lstm-w112-s1-cur.pt" --lengths 2048,8192,32768 --n 100 --batch 20 --out "$OUT/long-eval-recurrent.json" > "$OUT/long-eval-recurrent.txt" 2>&1
  if [ -f "$OUT/long-eval-recurrent.json" ]; then log "BITTI E-X recurrent (bkz $OUT/long-eval-recurrent.txt)"; else log "HATA E-X recurrent: sonuc dosyasi yok (bkz $OUT/long-eval-recurrent.txt)"; fi
fi
if [ ! -f "$OUT/long-eval-core.json" ]; then
  log "BASLA E-X (CPU, 1 parcacik): uzun uzunluk degerlendirmesi, S4 Core w64, 2048-32768 konum, n = 30"
  $PY -B -m horizon.long_eval "$OUT/perm4-core-w64-s1-cur.pt" --core-lengths 2048,8192,32768 --core-n 30 --batch 10 --out "$OUT/long-eval-core.json" > "$OUT/long-eval-core.txt" 2>&1
  if [ -f "$OUT/long-eval-core.json" ]; then log "BITTI E-X core (bkz $OUT/long-eval-core.txt)"; else log "HATA E-X core: sonuc dosyasi yok (bkz $OUT/long-eval-core.txt)"; fi
fi
