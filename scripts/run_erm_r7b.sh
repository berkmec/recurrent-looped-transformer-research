#!/usr/bin/env bash
# E-RM0R7b (G0.3b, notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.62): E-RM0R7's gate with the read's output projection
# started at zero (C_ds0, the start the language-scale warm start will use), seed 1, 6,000 updates; E-RM0R7's C_none is the control.
# Waits for E-RM0R7's queue to end; one thread, BelowNormal, unbuffered log, a RUNMAN manifest.
# Usage: nohup bash scripts/run_erm_r7b.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ERM-R7
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.62"
log() { echo "[$(date +%H:%M:%S)] [ermr7b] $*" >> "$LOG"; }

log "E-RM0R7b bekliyor: E-RM0R7 kuyrugu bitince baslar"
until grep -q -F "[ermr7] E-RM0R7 kuyrugu BITTI" "$LOG"; do sleep 60; done
id=Cds0-s1
if [ ! -f "$OUT/$id.json" ]; then
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "E-RM0R7b-$id" ERM-R7 claude-4797b78c "python -B -m horizon.erm_r train --arm C_ds0 --steps 6000 --seed 1 --out $OUT/$id.json" "$P" '{"seed": 1}' "$0" >> "$LOG" 2>&1 \
    || log "MANIFEST HATA $id"
  log "BASLA E-RM0R7b $id (C_ds0, tohum 1, 6000 guncelleme, muhurlu Core, Wo sifirdan)"
  "$PY" -B -u -m horizon.erm_r train --arm C_ds0 --steps 6000 --seed 1 --out "$OUT/$id.json" > "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI E-RM0R7b $id"; else log "HATA E-RM0R7b $id (bkz $OUT/$id.log)"; fi
fi
log "E-RM0R7b okuma: $("$PY" -B -m horizon.erm_r --reading "$OUT/$id.json" "$OUT/Cnone-s1.json" 2>/dev/null | "$PY" -B -c "import json, sys; r = json.load(sys.stdin); print(r.get('core_gate_zero_out'), r.get('acc_d3'))" 2>/dev/null)"
log "E-RM0R7b kuyrugu BITTI"
