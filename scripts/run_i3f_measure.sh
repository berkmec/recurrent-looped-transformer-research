#!/usr/bin/env bash
# I3-f measurement (rule: notes/_hive_F2_ONKAYIT_I3-f.md section 5, E-N protocol unchanged; no training). Waits until one copy of
# scripts/run_queue_2026-10-22b.sh has finished its list, so no more than two CPU experiment processes run at a time, then:
#   1. horizon.neutral_runs on the five cells (E-N protocol: length 512, n = 100, p_id 0 / 0.5 / 0.9) -> results/F2-I3f-en.json
#   2. horizon.hold on the register cells F2, F3, F5 (hold_neutral, sign_ok)                         -> results/F2-I3f-eg.json
# Usage: nohup bash scripts/run_i3f_measure.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] [i3f] $*" >> "$LOG"; }
W=results/F2-I3f
CELLS="$W-F1-perm4-core-w64-s1-np09.pt $W-F2-perm4-hold-w64-s1-np09.pt $W-F3-perm4-hold_nc-w64-s1-np09.pt $W-F4-perm5-core-w128-s1-np09.pt $W-F5-perm5-hold-w128-s1-np09.pt"
HOLD="$W-F2-perm4-hold-w64-s1-np09.pt $W-F3-perm4-hold_nc-w64-s1-np09.pt $W-F5-perm5-hold-w128-s1-np09.pt"
for f in $CELLS; do [ -f "$f" ] || { log "HATA I3-f olcum: agirlik yok $f"; exit 1; }; done

log "I3-f olcum bekliyor: bir q22b kopyasinin bitisi"
until grep -q "\[q22b\] kopya .* listeyi bitirdi" "$LOG"; do sleep 30; done

"$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
  F2-I3f-measure F2-I3f claude-4797b78c "python -B -m horizon.neutral_runs $CELLS --out $W-en.json; python -B -m horizon.hold $HOLD --out $W-eg.json" \
  "notes/_hive_F2_ONKAYIT_I3-f.md 5" '{"seed": 1}' "$0" >> "$LOG" 2>&1 || log "MANIFEST HATA F2-I3f-measure"
log "BASLA I3-f olcum (E-N protokolu, egitimsiz)"
if [ ! -f "$W-en.json" ]; then
  $PY -B -m horizon.neutral_runs $CELLS --out "$W-en.json" > "$W-en.txt" 2>&1 && log "BITTI I3-f E-N olcumu" || log "HATA I3-f E-N olcumu (bkz $W-en.txt)"
fi
if [ ! -f "$W-eg.json" ]; then
  $PY -B -m horizon.hold $HOLD --out "$W-eg.json" > "$W-eg.txt" 2>&1 && log "BITTI I3-f yazmac telemetrisi" || log "HATA I3-f yazmac telemetrisi (bkz $W-eg.txt)"
fi
