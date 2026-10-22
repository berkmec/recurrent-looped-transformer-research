#!/usr/bin/env bash
# E-RM0R5 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.54): the delta-rule associative memory arm B_delta (read gate
# started open, positions) on the repaired toy task, seeds 1 and 2 in parallel. One thread each, BelowNormal, RUNMAN manifest per run.
# Usage: nohup bash scripts/run_erm_r5.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ERM-R5
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.54"
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] [ermr5] $*" >> "$LOG"; }

run() {  # seed
  local id="Bdelta-s$1"
  [ -f "$OUT/$id.json" ] && return 0
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "E-RM0R5-$id" ERM-R5 claude-4797b78c "python -B -m horizon.erm_r train --arm B_delta --steps 12000 --seed $1 --positions --out $OUT/$id.json" "$P" "{\"seed\": $1}" "$0" >> "$LOG" 2>&1 \
    || log "MANIFEST HATA $id"
  log "BASLA E-RM0R5 $id (B_delta, delta kurali, gamma0 1, konumlu, 12000 guncelleme)"
  "$PY" -B -u -m horizon.erm_r train --arm B_delta --steps 12000 --seed "$1" --positions --out "$OUT/$id.json" > "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI E-RM0R5 $id"; else log "HATA E-RM0R5 $id (bkz $OUT/$id.log)"; fi
}

run 1 & run 2 & wait
log "E-RM0R5 okuma: s1 $("$PY" -B -m horizon.erm_r --reading results/ERM-R2/Apr-s1.json results/ERM-R2/A-s1.json "$OUT/Bdelta-s1.json" 2>/dev/null | tr -d '\n ' | cut -c1-220)"
log "E-RM0R5 okuma: s2 $("$PY" -B -m horizon.erm_r --reading results/ERM-R2/Apr-s1.json results/ERM-R2/A-s1.json "$OUT/Bdelta-s2.json" 2>/dev/null | tr -d '\n ' | cut -c1-220)"
log "E-RM0R5 kuyrugu BITTI"
