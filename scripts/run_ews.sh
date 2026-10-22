#!/usr/bin/env bash
# E-WS (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.48): feedback switched on after the full no-feedback run. From A0F's
# final weights (update 30,518), the same 2,000-update schedule on the same batches (data offset 30,518): (a) a0-cont, feedback off;
# (b) fb-ft, feedback on (RLT-1, CUDA graphs). Battery every 500. GPU, one run at a time. Reading: python -B -m mline.ews
# Usage: nohup bash scripts/run_ews.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."
export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
R=results/EWS
LOG=results/M1-finetune-distance/queue.log
SRC=results/L1-a0-w928-s401/latest.pt
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.48"
log() { echo "[$(date +%H:%M:%S)] [ews] $*" >> "$LOG"; }
STEP=$("$PY" -B -c "import torch; print(torch.load('$SRC', map_location='cpu', weights_only=False)['step'])" 2>/dev/null || echo ?)
if [ "$STEP" != 30518 ]; then log "HATA E-WS: A0F kontrol noktasi adimi $STEP (30518 bekleniyordu)"; exit 1; fi
COMMON=(--recipe m2-draft --tokens 32768000 --lr 2e-4 --warmup 100 --decay-fraction 0.1 --eval-every 500 --init-from "$SRC" --data-offset 30518)

run() {  # name arm [extra]
  local name="$1" arm="$2"; shift 2
  local d="$R/$name"
  [ -f "$d/complete.json" ] && return 0
  mkdir -p "$d"
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "E-WS-$name" EWS claude-4797b78c "python -B -m mline.longrun --out $d --arm $arm ${COMMON[*]} $*" "$P" '{"seed": 401}' "$0" >> "$LOG" 2>&1 \
    || log "MANIFEST HATA E-WS $name"
  log "BASLA E-WS $name (arm $arm $*; 2.000 guncelleme, A0F agirliklarindan)"
  "$PY" -B -m mline.longrun --out "$d" --arm "$arm" "${COMMON[@]}" "$@" >> "$d/train.log" 2>&1
  if [ -f "$d/complete.json" ]; then log "BITTI E-WS $name"; return 0; fi
  log "HATA E-WS $name (bkz $d/train.log)"; return 1
}

run a0-cont a0 && run fb-ft plain --graphs
"$PY" -B -m mline.ews --cont "$R/a0-cont" --fb "$R/fb-ft" --out "$R/reading.json" > "$R/reading.txt" 2>&1 \
  && log "BITTI E-WS okuma: $("$PY" -B -c "import json; r=json.load(open('$R/reading.json')); print(r['verdict'], 'delta2000', r.get('delta_last'))" 2>/dev/null)" \
  || log "HATA E-WS okuma (bkz $R/reading.txt)"
