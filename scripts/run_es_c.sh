#!/usr/bin/env bash
# E-S-C (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.58): the sk4 arms that failed at 15,000 updates, given about the
# Core's CPU wall-clock instead: GRU w128 at 150,000 updates, LSTM w112 at 165,000, the no-feedback twin at 75,000, the positional transformer
# at 60,000 (seed 1; a cell that learns gets seed 2 before the verdict). Waits until E-RM0R7 (G0.3', note 5.60, on the critical path of
# the GPU's stage 1) has ended so the CPU is not oversubscribed; three lanes, one thread each, BelowNormal, RUNMAN manifest per run.
# Usage: nohup bash scripts/run_es_c.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ES-compute
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.58"
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] [esc] $*" >> "$LOG"; }

log "E-S-C bekliyor: E-RM0R7 kuyrugu bitince baslar"
until grep -q -F "[ermr7] E-RM0R7 kuyrugu BITTI" "$LOG"; do sleep 60; done

job() {   # module arm width seed updates lr
  local mod="$1" arm="$2" w="$3" s="$4" u="$5" lr="$6" args
  local id="sk4-$arm-w$w-s$s-u$u"
  [ -f "$OUT/$id.json" ] && return 0
  mkdir "$OUT/$id.lock" 2>/dev/null || return 0
  if [ "$mod" = sk_pe ]; then args="--task sk4 --width $w --seed $s --lr $lr --steps $u --log-every 5000 --out $OUT/$id.json"
  else args="--task sk4 --arm $arm --width $w --seed $s --lr $lr --steps $u --log-every 5000 --out $OUT/$id.json"; fi
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "$id" ES-compute claude-4797b78c "python -B -m horizon.$mod $args" "$P" "{\"seed\": $s}" "$0" >> "$LOG" 2>&1 || log "MANIFEST HATA $id"
  log "BASLA E-S-C $id (CPU, 1 parcacik, $u adim, tepe lr $lr)"
  "$PY" -B -m "horizon.$mod" $args >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI E-S-C $id"; else log "HATA E-S-C $id (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

lane_a() { job sk_run gru 128 1 150000 0.003; }
lane_b() { job sk_run lstm 112 1 165000 0.003; job sk_run core_nofb 64 1 75000 0.003; }
lane_c() { job sk_pe tfm_pe 64 1 60000 0.001; }
lane_a & lane_b & lane_c & wait
log "E-S-C okuma: $("$PY" -B -m horizon.sk_compute $OUT/sk4-*.json 2>/dev/null | tr -d '\n' | tr -s ' ' | cut -c1-600)"
log "E-S-C kuyrugu BITTI"
