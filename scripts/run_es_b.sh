#!/usr/bin/env bash
# E-S: the tenth stage-A run (sk3 Core seed 2, cut on 2026-10-02) and stage B (sk4: 24 facts / 72 bits, five arms, seed 1).
# Reading rule and stage-B condition written before any run: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.16
# (stage B is triggered: the Core learned sk3 in seed 1). Same settings as stage A: 15,000 updates, batch 64, lr 0.003 annealed
# to ~0 over the last 30 % for every arm, walk curriculum 0..16, evaluation to walk 512. DEV split.
# One CPU process at a time, BelowNormal, one thread, idempotent (skips finished jobs; an atomic lock directory per job);
# RUNMAN manifest per run. The Core runs last (sk3 about 1.2 h, sk4 about 3 h).
#
# Usage: nohup bash scripts/run_es_b.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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
  local cmd="python -B -m horizon.sk_run --task $1 --arm $2 --width $3 --seed $4 --steps 15000 --out $OUT/$id.json --log-every 1000"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "$id" ES-pilot claude-4797b78c "$cmd" "notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.16" "{\"seed\": $4}" "$0" >> "$LOG" 2>&1 \
    || log "MANIFEST HATA $id"
  log "BASLA ES $id (CPU, 1 parcacik, 15000 adim, lr 0.003 son %30'da ~0'a, mufredat 0..16)"
  $PY -B -m horizon.sk_run --task "$1" --arm "$2" --width "$3" --seed "$4" --steps 15000 --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI ES $id"; else log "HATA ES $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

job sk3 core 64 2                      # stage A, tenth run: completes the registered stage-A verdict
job sk4 lstm 112 1
job sk4 gru 128 1
job sk4 core_nofb 64 1
job sk4 tfm 64 1
job sk4 core 64 1
log "ES kuyruk B BITTI"
