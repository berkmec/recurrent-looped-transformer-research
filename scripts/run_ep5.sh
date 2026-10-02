#!/usr/bin/env bash
# E-P5: is the S4 Core's E-X result (CLEAN: 0 first errors in 1,198,080 sequence-steps beyond 1,024 positions, all 30 sequences exact at 32,768) matched by an annealed
# PARAMETER-MATCHED GRU / LSTM? E-P and E-X annealed only the width-64 recurrent arms (28k / 36k parameters against the Core's 98k); here GRU w128 (105k) and LSTM w112 (107k)
# get the same schedule (lr 0.003, decayed to ~0 over the last 30 %, seed 1, 15,000 updates, curriculum 1..16) and the same long evaluation (2,048 / 8,192 / 32,768 positions, n = 100).
# Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.17. DEV split; NOT the frozen cycle.
# One CPU process, BelowNormal, idempotent.
#
# Usage: nohup bash scripts/run_ep5.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

CUR=1,2,3,4,6,8,12,16
job() {   # arm width
  local id="perm4-$1-w$2-s1-cur-lrdecay0.3"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (CPU, 1 parcacik, 15000 adim, lr 0.003 son %30'da ~0'a, mufredat $CUR; parametre-esit)"
  $PY -B -m horizon.depth_run --task perm4 --arm "$1" --width "$2" --seed 1 --steps 15000 --lr 0.003 --curriculum "$CUR" --lr-decay-fraction 0.3 --out "$OUT/$id.json" --log-every 2000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; fi
}
job gru 128
job lstm 112

if [ -f "$OUT/perm4-gru-w128-s1-cur-lrdecay0.3.pt" ] && [ -f "$OUT/perm4-lstm-w112-s1-cur-lrdecay0.3.pt" ] && [ ! -f "$OUT/long-eval-matched.json" ]; then
  log "BASLA E-P5 (CPU, 1 parcacik): uzun uzunluk degerlendirmesi, lr azaltmali GRU w128 ve LSTM w112, 2048-32768 konum"
  $PY -B -m horizon.long_eval "$OUT/perm4-gru-w128-s1-cur-lrdecay0.3.pt" "$OUT/perm4-lstm-w112-s1-cur-lrdecay0.3.pt" --lengths 2048,8192,32768 --n 100 --batch 20 \
    --out "$OUT/long-eval-matched.json" > "$OUT/long-eval-matched.txt" 2>&1
  if [ -f "$OUT/long-eval-matched.json" ]; then log "BITTI E-P5 (bkz $OUT/long-eval-matched.txt)"; else log "HATA E-P5: sonuc dosyasi yok"; fi
fi
log "Z1 kuyruk EP5 BITTI"
