#!/usr/bin/env bash
# E-G: chrono HOLD register + neutral-interval S4/S5 (R-3 step 1, user-approved 2026-10-14).
# Reading rule (registered BEFORE the runs): notes/EG_ONKAYIT_2026-10-14.md
# (M-1: hold >= 0.5 AND neutral-loss <= 5 pts -> DESTEKLENIR; hold < 0.1 or loss > 20 -> CURUTULDU).
# Reference arms (perm4-core / perm5-core) already exist on disk: not re-run.
# One CPU process at a time, BelowNormal, idempotent (skips finished jobs; an atomic lock
# directory per job lets two queue chains share it). ~5 CPU-h total (S4 ~35 dk/run, S5 ~90 dk/run).
#
# Usage: nohup bash scripts/run_eg.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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
manifest() {  # RUNMAN tek-yazim manifest (_hive_KOSU_HIJYENI_SEMA.md; god 7060c0)
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "$1" "$2" jim-mv8403zi "$3" "$4" "$5" "$0" >> "$LOG" 2>&1 || log "MANIFEST HATA $1"
}

CUR=1,2,3,4,6,8,12,16
job() {   # task arm width seed steps
  local id="$1-$2-w$3-s$4-cur-egnp05"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  manifest "$id" Z1-pilot "python -B -m horizon.depth_run --task $1 --arm $2 --width $3 --seed $4 --steps $5 --lr 0.003 --curriculum $CUR --neutral-p 0.5 --hold-tmax 16" "notes/EG_ONKAYIT_2026-10-14.md" "{\"init\":$4,\"data_split\":$4}"
  log "BASLA EG $id (CPU, 1 parcacik, $5 adim, mufredat $CUR, notr-p 0.5, hold-tmax 16, kapida sonum yok)"
  $PY -B -m horizon.depth_run --task "$1" --arm "$2" --width "$3" --seed "$4" --steps "$5" --lr 0.003 --curriculum "$CUR" --neutral-p 0.5 --hold-tmax 16 --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI EG $id"; else log "HATA EG $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

job perm4 hold 64 1 15000
job perm4 hold 64 2 15000
job perm4 hold 64 3 15000
job perm4 hold_nc 64 1 15000
job perm5 hold 128 1 30000
job perm5 hold 128 2 30000
log "EG kuyruk BITTI"
