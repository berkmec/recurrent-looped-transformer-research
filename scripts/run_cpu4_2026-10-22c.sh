#!/usr/bin/env bash
# CPU queue 2026-10-22 c (user 21:3x: "hesaplama gücünü yığ, bilgisayarla işim yok, OOM dikkat et"; four CPU processes, memory
# measured after each start): E-RM0R4 (note 5.53: B_orc and B with the read gate started at 1, positions) and E-S-PE (note 5.52:
# the E-S transformer with sinusoidal positions on sk4, lr 1e-3 / 3e-3, seeds 1-2). Work-stealing: start FOUR copies; each walks the
# same list and takes the first job with neither a result nor a lock. One thread per process, BelowNormal, a RUNMAN manifest per run.
# Usage: nohup bash scripts/run_cpu4_2026-10-22c.sh > /dev/null 2>&1 &   (four times, one at a time; progress: queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=results/M1-finetune-distance/queue.log
P=notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md
mkdir -p results/ERM-R4 results/ES-pe
log() { echo "[$(date +%H:%M:%S)] [q22c] $*" >> "$LOG"; }
manifest() {  # run_id campaign command prereg seeds
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "$1" "$2" claude-4797b78c "$3" "$4" "$5" "$0" >> "$LOG" 2>&1 || log "MANIFEST HATA $1"
}

erm() {  # id arm
  local d=results/ERM-R4 id="$1" arm="$2"
  [ -f "$d/$id.json" ] && return 0
  mkdir "$d/$id.lock" 2>/dev/null || return 0
  manifest "E-RM0R4-$id" ERM-R4 "python -B -m horizon.erm_r train --arm $arm --steps 12000 --seed 1 --positions --gamma-init 1.0 --out $d/$id.json" "$P 5.53" '{"seed": 1}'
  log "BASLA E-RM0R4 $id ($arm, gamma0 1.0, konumlu, 12000 guncelleme)"
  "$PY" -B -m horizon.erm_r train --arm "$arm" --steps 12000 --seed 1 --positions --gamma-init 1.0 --out "$d/$id.json" > "$d/$id.log" 2>&1
  if [ -f "$d/$id.json" ]; then log "BITTI E-RM0R4 $id"; else log "HATA E-RM0R4 $id (bkz $d/$id.log)"; rmdir "$d/$id.lock" 2>/dev/null; fi
}

pe() {  # seed lr
  local d=results/ES-pe id="sk4-tfm_pe-w64-s$1-lr$2"
  [ -f "$d/$id.json" ] && return 0
  mkdir "$d/$id.lock" 2>/dev/null || return 0
  manifest "$id" ES-pe "python -B -m horizon.sk_pe --task sk4 --width 64 --seed $1 --lr $2 --steps 15000 --out $d/$id.json" "$P 5.52" "{\"seed\": $1}"
  log "BASLA ES $id (konumlu transformer, 15000 adim, tepe lr $2)"
  "$PY" -B -m horizon.sk_pe --task sk4 --width 64 --seed "$1" --lr "$2" --steps 15000 --out "$d/$id.json" > "$d/$id.log" 2>&1
  if [ -f "$d/$id.json" ]; then log "BITTI ES $id"; else log "HATA ES $id (bkz $d/$id.log)"; rmdir "$d/$id.lock" 2>/dev/null; fi
}

erm Borc-g1-s1 B_orc
erm B-g1-s1 B
pe 1 0.001
pe 1 0.003
pe 2 0.001
pe 2 0.003
log "kopya $$ listeyi bitirdi (q22c)"
