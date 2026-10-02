#!/usr/bin/env bash
# E-W: weight-space margin of the trained S4 arms. Every weight tensor w gets w + eps x rms(w) x N(0, 1) (eps 0 ... 0.1, 3 independent draws per eps), then the standard dev
# evaluation (lengths 16 / 64 / 256 / 1,024, n = 100) gives the pooled first-error hazard; eps_tol = the smallest eps with hazard >= 1e-3. Models: the S4 Core w64 (constant lr) and, for the
# GRU w64 / LSTM w64 / GRU w128 / LSTM w112, the constant-lr run and its annealed twin (E-P, E-P5). Evaluation only. Reading rule written before this ran:
# notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.18. One CPU process, BelowNormal, idempotent.
#
# Usage: nohup bash scripts/run_ew.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ ! -f "$OUT/weight-noise-recurrent.json" ]; then
  MODELS=""
  for m in perm4-gru-w64-s1-cur perm4-gru-w64-s1-cur-lrdecay0.3 perm4-lstm-w64-s1-cur perm4-lstm-w64-s1-cur-lrdecay0.3 perm4-gru-w128-s1-cur perm4-gru-w128-s1-cur-lrdecay0.3 \
           perm4-lstm-w112-s1-cur perm4-lstm-w112-s1-cur-lrdecay0.3; do
    if [ -f "$OUT/$m.pt" ]; then MODELS="$MODELS $OUT/$m.pt"; fi
  done
  log "BASLA E-W (CPU, 1 parcacik): agirlik gurultusu marji, GRU/LSTM (sabit ve lr azaltmali)"
  $PY -B -m horizon.weight_noise $MODELS --draws 3 --out "$OUT/weight-noise-recurrent.json" > "$OUT/weight-noise-recurrent.txt" 2>&1
  if [ -f "$OUT/weight-noise-recurrent.json" ]; then log "BITTI E-W recurrent (bkz $OUT/weight-noise-recurrent.txt)"; else log "HATA E-W recurrent: sonuc dosyasi yok"; fi
fi
if [ ! -f "$OUT/weight-noise-core.json" ]; then
  log "BASLA E-W (CPU, 1 parcacik): agirlik gurultusu marji, S4 Core w64"
  $PY -B -m horizon.weight_noise "$OUT/perm4-core-w64-s1-cur.pt" --draws 3 --out "$OUT/weight-noise-core.json" > "$OUT/weight-noise-core.txt" 2>&1
  if [ -f "$OUT/weight-noise-core.json" ]; then log "BITTI E-W core (bkz $OUT/weight-noise-core.txt)"; else log "HATA E-W core: sonuc dosyasi yok"; fi
fi
