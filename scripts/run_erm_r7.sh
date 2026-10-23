#!/usr/bin/env bash
# E-RM0R7 (G0.3', notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.60): the sealed Core with the RLT-M memory (mline.rltm) on
# the toy distance task: C_ds seeds 1-2 and C_none seed 1, 6,000 updates; C_full (the positive control) only if a C_ds seed misses
# the gate. Waits for E-RM0R6's three 12,000-update runs so the CPU is not oversubscribed; one thread per run, BelowNormal,
# unbuffered logs, a RUNMAN manifest per run.
# Usage: nohup bash scripts/run_erm_r7.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ERM-R7
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.60"
STEPS=6000
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] [ermr7] $*" >> "$LOG"; }

log "E-RM0R7 bekliyor: E-RM0R6 Bshift-s1, Bshift-s2, Borc-g1-s2 bitince baslar"
for id in Bshift-s1 Bshift-s2 Borc-g1-s2; do
  until grep -q -E "\[ermr6\] (BITTI|HATA) E-RM0R6 $id" "$LOG"; do sleep 60; done
done

run() {  # id arm seed
  local id="$1" arm="$2" seed="$3"
  [ -f "$OUT/$id.json" ] && return 0
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "E-RM0R7-$id" ERM-R7 claude-4797b78c "python -B -m horizon.erm_r train --arm $arm --steps $STEPS --seed $seed --out $OUT/$id.json" "$P" "{\"seed\": $seed}" "$0" >> "$LOG" 2>&1 \
    || log "MANIFEST HATA $id"
  log "BASLA E-RM0R7 $id ($arm, tohum $seed, $STEPS guncelleme, muhurlu Core)"
  "$PY" -B -u -m horizon.erm_r train --arm "$arm" --steps "$STEPS" --seed "$seed" --out "$OUT/$id.json" > "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI E-RM0R7 $id"; else log "HATA E-RM0R7 $id (bkz $OUT/$id.log)"; fi
}
gate() {  # C_ds seed, then the other result files
  local s="$1"; shift
  "$PY" -B -m horizon.erm_r --reading "$OUT/Cds-s$s.json" "$@" 2>/dev/null | "$PY" -B -c "import json, sys; print(json.load(sys.stdin).get('core_gate'))" 2>/dev/null
}

run Cds-s1 C_ds 1 & run Cds-s2 C_ds 2 & run Cnone-s1 C_none 1 & wait
g1=$(gate 1 "$OUT/Cnone-s1.json"); g2=$(gate 2 "$OUT/Cnone-s1.json")
log "E-RM0R7 okuma: s1 $g1 | s2 $g2"
if [ "$g1" != "GATE PASSED" ] || [ "$g2" != "GATE PASSED" ]; then
  run Cfull-s1 C_full 1
  log "E-RM0R7 okuma (C_full ile): s1 $(gate 1 "$OUT/Cnone-s1.json" "$OUT/Cfull-s1.json") | s2 $(gate 2 "$OUT/Cnone-s1.json" "$OUT/Cfull-s1.json")"
fi
log "E-RM0R7 kuyrugu BITTI"
