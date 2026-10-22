#!/usr/bin/env bash
# E-RM0R2 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.49): the repaired toy E-RM task with sinusoidal positions in the encoder (--positions); same ladder as E-RM0R (5.46).
# Waits until both copies of scripts/run_es_b2x.sh have finished (at most two CPU experiment processes), then R1 (Apr, A in parallel)
# and, by the registered rule, R2 (B, B_comp) or R1b (Apr with a facts
# curriculum) -> R2b (A, B, B_comp with the curriculum).
# Usage: nohup bash scripts/run_erm_r2.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ERM-R2
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.49"
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] [ermr2] $*" >> "$LOG"; }

log "E-RM0R2 bekliyor: iki b2x kopyasi"
until [ "$(grep -c 'listeyi bitirdi (b2x)' "$LOG")" -ge 2 ]; do sleep 60; done

run() {  # name arm steps [curriculum]
  local id="$1" arm="$2" steps="$3" cur="${4:-}"
  [ -f "$OUT/$id.json" ] && return 0
  local extra=(); [ -n "$cur" ] && extra=(--curriculum "$cur")
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "E-RM0R2-$id" ERM-R2 claude-4797b78c "python -B -m horizon.erm_r train --arm $arm --steps $steps --seed 1 --positions ${extra[*]:-} --out $OUT/$id.json" "$P" '{"seed": 1}' "$0" >> "$LOG" 2>&1 \
    || log "MANIFEST HATA $id"
  log "BASLA E-RM0R2 $id ($arm, $steps guncelleme${cur:+, mufredat $cur})"
  "$PY" -B -m horizon.erm_r train --arm "$arm" --steps "$steps" --seed 1 --positions ${extra[@]+"${extra[@]}"} --out "$OUT/$id.json" > "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI E-RM0R2 $id"; else log "HATA E-RM0R2 $id (bkz $OUT/$id.log)"; fi
}
verdict() { "$PY" -B -m horizon.erm_r --reading "$@" 2>/dev/null; }

run Apr-s1 Apr 12000 & run A-s1 A 12000 & wait
R1=$(verdict "$OUT/Apr-s1.json" "$OUT/A-s1.json")
log "E-RM0R2 R1 okuma: $(echo "$R1" | tr -d '\n ' | cut -c1-200)"
if echo "$R1" | grep -q '"LEARNABLE"' && echo "$R1" | grep -q '"CONTROL OK"'; then
  run B-s1 B 12000 & run Bcomp-s1 B_comp 12000 & wait
  log "E-RM0R2 R2 okuma: $(verdict "$OUT/Apr-s1.json" "$OUT/A-s1.json" "$OUT/B-s1.json" "$OUT/Bcomp-s1.json" | tr -d '\n ' | cut -c1-260)"
elif echo "$R1" | grep -q '"NOT LEARNABLE"'; then
  run Apr-cur-s1 Apr 24000 4,8,16
  R1b=$(verdict "$OUT/Apr-cur-s1.json")
  log "E-RM0R2 R1b okuma: $(echo "$R1b" | tr -d '\n ' | cut -c1-200)"
  if echo "$R1b" | grep -q '"LEARNABLE"'; then
    run A-cur-s1 A 24000 4,8,16 & run B-cur-s1 B 24000 4,8,16 & wait
    run Bcomp-cur-s1 B_comp 24000 4,8,16
    log "E-RM0R2 R2b okuma: $(verdict "$OUT/Apr-cur-s1.json" "$OUT/A-cur-s1.json" "$OUT/B-cur-s1.json" "$OUT/Bcomp-cur-s1.json" | tr -d '\n ' | cut -c1-260)"
  fi
fi
log "E-RM0R2 kuyrugu BITTI"
