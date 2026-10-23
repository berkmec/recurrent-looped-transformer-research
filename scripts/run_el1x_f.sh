#!/usr/bin/env bash
# E-L1X-F (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.66): does the trained L1's carried state bring context from beyond the 512-position
# attention window? mline.state_reach on the final weights (30,518 updates, latest.pt: the registered verdict), then the 24k / 18k / 12k / 6k snapshots
# (the curve), then the final weights on 16 articles x 8,192 bytes (descriptive). Evaluation only, CPU, one process, BelowNormal; starts once G0.4b has
# been measured (so the stage-1 gate sees the CPU load of the first measurement); idempotent.
# Usage: nohup bash scripts/run_el1x_f.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=results/L1-scratch-w928-s401
OUT=results/L1-long-doc
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.66"
log() { echo "[$(date +%H:%M:%S)] [l1xf] $*" >> "$LOG"; }
mkdir -p "$OUT"

log "E-L1X-F bekliyor: G0.4b olculunce baslar (CPU)"
until grep -q -E "\[erm1b\] (G0.4b|HATA Asama 1|Asama 1 DURDU)" "$LOG"; do sleep 60; done
"$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
  E-L1X-F-L1-scratch-w928-s401 L1-long-doc claude-4797b78c "python -B -m mline.state_reach <latest.pt | snapshots> --out $OUT [--length 8192 --n 16]" "$P" '{"seed": 401}' "$0" >> "$LOG" 2>&1 \
  || log "MANIFEST HATA E-L1X-F"

one() {  # weights, output name, length, articles
  local path="$1" name="$2" length="$3" n="$4" suffix=""
  if [ "$length" != "4096" ]; then suffix="-$length"; fi
  [ -f "$OUT/$name-state-reach$suffix.json" ] && return 0
  if [ ! -f "$path" ]; then log "HATA E-L1X-F $name: $path yok"; return 0; fi
  log "BASLA E-L1X-F $name ($length bayt, $n makale; win512 / winreset512 / chunk512; CPU)"
  "$PY" -B -m mline.state_reach "$path" --length "$length" --n "$n" --out "$OUT" > "$OUT/$name-state-reach$suffix.txt" 2>&1
  if [ -f "$OUT/$name-state-reach$suffix.json" ]; then
    log "BITTI E-L1X-F $name$suffix reading: $("$PY" -B -c "import json, sys; r = json.load(open(sys.argv[1], encoding='utf8')); print(r['reading']['verdict'], {k: [round(x, 4) for x in v] for k, v in r['reading'].get('s', {}).items()})" "$OUT/$name-state-reach$suffix.json" 2>/dev/null)"
  else
    log "HATA E-L1X-F $name$suffix (bkz $OUT/$name-state-reach$suffix.txt)"
  fi
}

one "$RUN/latest.pt" step030518 4096 32
for s in 024000 018000 012000 006000; do one "$RUN/snapshots/step$s.pt" "step$s" 4096 32; done
one "$RUN/latest.pt" step030518 8192 16
log "E-L1X-F kuyruk BITTI"
