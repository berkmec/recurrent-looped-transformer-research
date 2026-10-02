#!/usr/bin/env bash
# E-S stage A: state-keyed retrieval on S3 (sk3): five arms x seeds 1 and 2. Does the combination an RLT-style model claims (recurrent state tracked through a long walk + an attention
# memory of prefix facts keyed by that state) show up as an advantage over a recurrent-only arm (GRU, LSTM), a memory-only arm (the A0 Core without feedback, a NoPE transformer)?
# Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.16 (python -B -m horizon.sk_run --reading results/ES-pilot/sk3-*.json).
# 15,000 updates, batch 64, lr 0.003 annealed to ~0 over the last 30 % FOR EVERY ARM, walk curriculum 0..16, evaluation to walk 512. DEV split; NOT the frozen cycle.
# One CPU process at a time, BelowNormal, idempotent (skips finished jobs; an atomic lock directory per job lets two queue chains share it). Cheap arms first, the Core last
# (about 1.2 h per run).
#
# Usage: nohup bash scripts/run_es.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ES-pilot
LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

job() {   # task arm width seed
  local id="$1-$2-w$3-s$4"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA ES $id (CPU, 1 parcacik, 15000 adim, lr 0.003 son %30'da ~0'a, mufredat 0..16)"
  $PY -B -m horizon.sk_run --task "$1" --arm "$2" --width "$3" --seed "$4" --steps 15000 --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI ES $id"; else log "HATA ES $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

for seed in 1 2; do
  job sk3 lstm 112 "$seed"
  job sk3 gru 128 "$seed"
  job sk3 core_nofb 64 "$seed"
  job sk3 tfm 64 "$seed"
done
for seed in 1 2; do job sk3 core 64 "$seed"; done
log "ES kuyruk A BITTI"
