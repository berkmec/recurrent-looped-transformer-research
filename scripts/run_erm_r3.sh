#!/usr/bin/env bash
# E-RM0R3 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.51): where the toy memory fails - the oracle-write arm B_orc and
# the RMT-style arm B_rmt on the repaired task with positions; the positive control and the control are E-RM0R2's (same code,
# seed and recipe). Two CPU processes in parallel, one thread each, BelowNormal.
# Usage: nohup bash scripts/run_erm_r3.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ERM-R3
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.51"
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] [ermr3] $*" >> "$LOG"; }

run() {  # name arm
  local id="$1" arm="$2"
  [ -f "$OUT/$id.json" ] && return 0
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "E-RM0R3-$id" ERM-R3 claude-4797b78c "python -B -m horizon.erm_r train --arm $arm --steps 12000 --seed 1 --positions --out $OUT/$id.json" "$P" '{"seed": 1}' "$0" >> "$LOG" 2>&1 \
    || log "MANIFEST HATA $id"
  log "BASLA E-RM0R3 $id ($arm, 12000 guncelleme, konumlu)"
  "$PY" -B -m horizon.erm_r train --arm "$arm" --steps 12000 --seed 1 --positions --out "$OUT/$id.json" > "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI E-RM0R3 $id"; else log "HATA E-RM0R3 $id (bkz $OUT/$id.log)"; fi
}

run Borc-s1 B_orc & run Brmt-s1 B_rmt & wait
log "E-RM0R3 okuma: $("$PY" -B -m horizon.erm_r --reading results/ERM-R2/Apr-s1.json results/ERM-R2/A-s1.json "$OUT/Borc-s1.json" "$OUT/Brmt-s1.json" 2>/dev/null | tr -d '\n ' | cut -c1-300)"
log "E-RM0R3 kuyrugu BITTI"
