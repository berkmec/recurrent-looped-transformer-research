#!/usr/bin/env bash
# E-C-ROT (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.63): the counter of E-C (count8_32) with the plain Core and the
# Core plus a rotary counter channel (horizon.depth_rot: rounded steps core_rotq, tanh steps core_rot); width 64, seed 1, 12,000
# updates, lr 0.003, E-C's points. Each run starts only when a CPU lane is free (at most 8 experiment interpreters, L1 included),
# one thread, BelowNormal, a RUNMAN manifest per run; the registered reading at the end.
# Usage: nohup bash scripts/run_ec_rot.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.63"
POINTS=8,16,32,64,128,256,512
log() { echo "[$(date +%H:%M:%S)] [ecrot] $*" >> "$LOG"; }
running() {   # experiment interpreters (the venv launcher of each is not counted); an empty answer counts as full
  local n
  n=$(powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '-m (mline|horizon)\.' -and \$_.ExecutablePath -notmatch 'venv' }).Count" 2>/dev/null | tr -d '\r ')
  echo "${n:-99}"
}

job() {   # module arm
  local id="count8_32-$2-w64-s1" cmd
  [ -f "$OUT/$id.json" ] && return 0
  cmd="python -B -m horizon.$1 --task count8_32 --arm $2 --width 64 --seed 1 --steps 12000 --lr 0.003 --points $POINTS --out $OUT/$id.json --log-every 1000"
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" "E-C-ROT-$id" Z1-pilot claude-4797b78c "$cmd" "$P" '{"seed": 1}' "$0" >> "$LOG" 2>&1 \
    || log "MANIFEST HATA $id"
  log "BASLA E-C-ROT $id (CPU, 1 parcacik, 12000 adim)"
  "$PY" -B -m "horizon.$1" --task count8_32 --arm "$2" --width 64 --seed 1 --steps 12000 --lr 0.003 --points "$POINTS" --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI E-C-ROT $id"; else log "HATA E-C-ROT $id (bkz $OUT/$id.log)"; fi
}

log "E-C-ROT bekliyor: her kol bos bir CPU hatti acilinca baslar"
pids=()
for spec in "depth_rot core_rotq" "depth_run core" "depth_rot core_rot"; do
  until [ "$(running)" -lt 8 ]; do sleep 120; done
  job $spec &
  pids+=($!)
  sleep 90                               # let the new interpreter appear before counting again
done
wait "${pids[@]}"
log "E-C-ROT okuma: $("$PY" -B -m horizon.depth_rot --reading $OUT/count8_32-core-w64-s1.json $OUT/count8_32-core_rotq-w64-s1.json $OUT/count8_32-core_rot-w64-s1.json 2>/dev/null | "$PY" -B -c "import json, sys; r = json.load(sys.stdin); print(r['verdict'], {a: (v['valid'], v['rho']) for a, v in r['runs'].items()})" 2>/dev/null)"
log "E-C-ROT kuyrugu BITTI"
