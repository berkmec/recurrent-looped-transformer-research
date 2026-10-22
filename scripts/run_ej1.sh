#!/usr/bin/env bash
# E-J1 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.44): the feedback Core's training step replayed as CUDA graphs, at
# L1 scale, on a FREE GPU (waits for E-A0F to end). From L1's last checkpoint (update 8,500; copied: L1's own directory is not
# touched), two eager copies and one graphed copy run updates 8,501-8,520 on the full schedule; the graphed copy then goes on to
# 9,000 so the battery runs while the graphs' memory pool is held. Telemetry std1 in every copy (per-update wall time; the same
# extra memory everywhere). GPU memory is sampled every 5 s. Reading: python -B -m mline.graphed results/EJ1-graphs
# Usage: nohup bash scripts/run_ej1.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."
export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
R=results/EJ1-graphs
LOG=results/M1-finetune-distance/queue.log
SRC=results/L1-scratch-w928-s401/latest.pt
P=notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md
log() { echo "[$(date +%H:%M:%S)] [ej1] $*" >> "$LOG"; }

log "E-J1 bekliyor: E-A0F bitisi (bos GPU)"
until grep -q -E "BITTI E-A0F|HATA E-A0F" "$LOG"; do sleep 30; done
mkdir -p "$R"
STEP=$("$PY" -B -c "import torch; print(torch.load('$SRC', map_location='cpu', weights_only=False)['step'])" 2>/dev/null || echo ?)
if [ "$STEP" != 8500 ]; then log "HATA E-J1: L1 kontrol noktasi adimi $STEP (8500 bekleniyordu)"; exit 1; fi
"$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
  E-J1-graphs EJ1-graphs claude-4797b78c "python -B -m mline.longrun --out $R/{eager1,eager2,graphed} --recipe m2-draft --tokens 500000000 --telemetry std1 --stop-at 8520 [--graphs]; graphed again --stop-at 9000 --graphs" \
  "$P 5.44" '{"seed": 401}' "$0" >> "$LOG" 2>&1 || log "MANIFEST HATA E-J1"

( while true; do echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> "$R/vram.txt"; sleep 5; done ) &
SAMPLER=$!

run() {  # name stop [--graphs]
  local d="$R/$1" stop="$2"; shift 2
  [ -f "$d/at$stop.pt" ] && return 0
  mkdir -p "$d"
  [ -f "$d/latest.pt" ] || cp "$SRC" "$d/latest.pt"
  log "BASLA E-J1 $(basename "$d") -> $stop $*"
  "$PY" -B -m mline.longrun --out "$d" --recipe m2-draft --tokens 500000000 --telemetry std1 --stop-at "$stop" "$@" >> "$d/train.log" 2>&1
  local got
  got=$("$PY" -B -c "import json; print(json.load(open('$d/stopped.json'))['stopped_at'])" 2>/dev/null || echo ?)
  if [ "$got" = "$stop" ]; then cp "$d/latest.pt" "$d/at$stop.pt"; log "BITTI E-J1 $(basename "$d") -> $stop"; return 0; fi
  log "HATA E-J1 $(basename "$d") -> $stop (bkz $d/train.log)"; return 1
}

run eager1 8520 && run eager2 8520 && run graphed 8520 --graphs && run graphed 9000 --graphs
kill "$SAMPLER" 2>/dev/null
"$PY" -B -m mline.graphed "$R" --out "$R/reading.json" > "$R/reading.txt" 2>&1 \
  && log "BITTI E-J1 okuma: $("$PY" -B -c "import json; r=json.load(open('$R/reading.json')); print(r['verdict'], 'hiz', r.get('speedup'))" 2>/dev/null)" \
  || log "HATA E-J1 okuma (bkz $R/reading.txt)"
