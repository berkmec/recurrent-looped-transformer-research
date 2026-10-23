#!/usr/bin/env bash
# E-S-W (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.57): sk4 with the GRU and LSTM baselines at about 4x and 16x the
# Core's parameters and the positional transformer at about 8x. Three CPU lanes, one thread each, BelowNormal, RUNMAN manifest per run.
# Usage: nohup bash scripts/run_es_w.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ES-width
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.57"
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] [esw] $*" >> "$LOG"; }

job() {   # module arm width seed lr
  local mod="$1" arm="$2" w="$3" s="$4" lr="$5" args
  local id="sk4-$arm-w$w-s$s-lr$lr"
  [ -f "$OUT/$id.json" ] && return 0
  mkdir "$OUT/$id.lock" 2>/dev/null || return 0
  if [ "$mod" = sk_pe ]; then args="--task sk4 --width $w --seed $s --lr $lr --steps 15000 --log-every 1000 --out $OUT/$id.json"
  else args="--task sk4 --arm $arm --width $w --seed $s --lr $lr --steps 15000 --log-every 1000 --out $OUT/$id.json"; fi
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "$id" ES-width claude-4797b78c "python -B -m horizon.$mod $args" "$P" "{\"seed\": $s}" "$0" >> "$LOG" 2>&1 || log "MANIFEST HATA $id"
  log "BASLA E-S-W $id (CPU, 1 parcacik, 15000 adim)"
  "$PY" -B -m "horizon.$mod" $args >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI E-S-W $id"; else log "HATA E-S-W $id (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

lane_gru()  { job sk_run gru 256 1 0.003; job sk_run gru 256 2 0.003; job sk_run gru 512 1 0.001; }
lane_lstm() { job sk_run lstm 224 1 0.003; job sk_run lstm 224 2 0.003; job sk_run lstm 448 1 0.001; }
lane_tfm()  { job sk_pe tfm_pe 128 1 0.001; }
lane_gru & lane_lstm & lane_tfm & wait
log "E-S-W okuma: $("$PY" -B -m horizon.sk_width $OUT/sk4-*.json 2>/dev/null | tr -d '\n' | tr -s ' ' | cut -c1-600)"
log "E-S-W kuyrugu BITTI"
