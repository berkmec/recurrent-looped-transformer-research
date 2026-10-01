#!/usr/bin/env bash
# Z1 pilot, second queue: the counter re-run with depth as the only extrapolated quantity (training length 64), then the hard permutation
# tasks (S_4: 24 states, S_5: 120 states) on the sealed Core. DEV split only; NOT the frozen cycle. Two lanes, one CPU thread each.
#
# Usage: nohup bash scripts/run_z1_pilot2.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."
PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

job() {   # task arm width seed steps
  local id="$1-$2-w$3-s$4"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  log "BASLA Z1-pilot $id (CPU, 1 parcacik, $5 adim)"
  $PY -B -m horizon.depth_run --task "$1" --arm "$2" --width "$3" --seed "$4" --steps "$5" --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1-pilot $id"; else log "HATA Z1-pilot $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; fi
}

( job count core 64 1 4000; job perm5 core 128 1 6000 ) &
( job count gru 64 1 4000; job perm4 core 64 1 4000; job perm4 gru 64 1 4000 ) &
wait
log "Z1-pilot kuyruk 2 BITTI"
