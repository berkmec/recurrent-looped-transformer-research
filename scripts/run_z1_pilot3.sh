#!/usr/bin/env bash
# Z1 pilot, third queue: the hard permutation tasks (S_5 120 states, S_4 Core which did not learn in 4,000 fixed-length updates) with a length CURRICULUM and a serious budget. The first pilot (4000
# updates, fixed length 16) did not get a GRU past loss 3.6 (chance 4.79): a budget problem, not an answer. DEV split only; NOT the frozen
# cycle. The curriculum only ever trains on lengths <= 16, so the extrapolation test is unchanged.
#
# Usage: nohup bash scripts/run_z1_pilot3.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."
PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
CUR=1,2,3,4,6,8,12,16
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

job() {   # task arm width seed steps lr
  local id="$1-$2-w$3-s$4-cur"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  log "BASLA Z1-pilot $id (CPU, 1 parcacik, $5 adim, mufredat $CUR, lr $6)"
  $PY -B -m horizon.depth_run --task "$1" --arm "$2" --width "$3" --seed "$4" --steps "$5" --lr "$6" --curriculum "$CUR" --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1-pilot $id"; else log "HATA Z1-pilot $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; fi
}

( job perm5 gru 256 1 30000 0.001 ) &
( job perm5 core 128 1 30000 0.003 ) &
( job perm4 core 64 1 15000 0.003 ) &
wait
log "Z1-pilot kuyruk 3 BITTI"
