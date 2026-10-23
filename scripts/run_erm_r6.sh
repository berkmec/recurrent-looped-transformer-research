#!/usr/bin/env bash
# E-RM0R6 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.56): the shifted-key delta memory (seeds 1, 2), the plain delta
# memory with a 3x budget (seed 1, 36,000 updates) and the oracle-write arm with the read gate open (seed 2). Four CPU processes in
# parallel, one thread each, BelowNormal, unbuffered logs, a RUNMAN manifest per run.
# Usage: nohup bash scripts/run_erm_r6.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ERM-R6
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.56"
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] [ermr6] $*" >> "$LOG"; }

run() {  # id arm seed steps [extra...]
  local id="$1" arm="$2" seed="$3" steps="$4"; shift 4
  [ -f "$OUT/$id.json" ] && return 0
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "E-RM0R6-$id" ERM-R6 claude-4797b78c "python -B -m horizon.erm_r train --arm $arm --steps $steps --seed $seed --positions $* --out $OUT/$id.json" "$P" "{\"seed\": $seed}" "$0" >> "$LOG" 2>&1 \
    || log "MANIFEST HATA $id"
  log "BASLA E-RM0R6 $id ($arm, tohum $seed, $steps guncelleme, konumlu $*)"
  "$PY" -B -u -m horizon.erm_r train --arm "$arm" --steps "$steps" --seed "$seed" --positions "$@" --out "$OUT/$id.json" > "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI E-RM0R6 $id"; else log "HATA E-RM0R6 $id (bkz $OUT/$id.log)"; fi
}

run Bshift-s1 B_delta_shift 1 12000 & run Bshift-s2 B_delta_shift 2 12000 & run Bdelta-s1-36k B_delta 1 36000 & run Borc-g1-s2 B_orc 2 12000 --gamma-init 1.0 & wait
for f in Bshift-s1 Bshift-s2 Bdelta-s1-36k Borc-g1-s2; do
  log "E-RM0R6 okuma $f: $("$PY" -B -m horizon.erm_r --reading results/ERM-R2/Apr-s1.json results/ERM-R2/A-s1.json "$OUT/$f.json" 2>/dev/null | tr -d '\n ' | cut -c1-200)"
done
log "E-RM0R6 kuyrugu BITTI"
