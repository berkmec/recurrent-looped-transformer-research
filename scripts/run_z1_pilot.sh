#!/usr/bin/env bash
# Z1 pilot: short exploratory runs of the depth-extrapolation suite (horizon/depth_run.py). NOT the frozen cycle: they find out whether each
# arm learns a task at all, how fast, and how far it extrapolates, on the DEV split, so the frozen protocol can be sized before any test
# data is opened. Three workers at a time, one CPU thread each (the GPU belongs to the long run).
#
# Usage: nohup bash scripts/run_z1_pilot.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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

( job perm3 core 64 1 4000 ) &
( job perm3 core_nofb 64 1 4000 ) &
( job count core 64 1 4000 ) &
wait
( job perm5 gru 128 1 4000 ) &
( job count gru 64 1 4000 ) &
( job perm3 gru 64 1 4000 ) &
wait
log "Z1-pilot dalga 1 BITTI"
