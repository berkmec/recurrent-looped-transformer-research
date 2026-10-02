#!/usr/bin/env bash
# E-P4: is the S4 GRU/LSTM constant-lr error floor TYPICAL (annealing removes it) or a seed lottery? Seeds 4 and 5, both arms, both conditions (constant lr and the E-P annealed
# schedule), width 64, 15,000 updates, curriculum 1,2,3,4,6,8,12,16, lr 0.003 (the seed-1 recipe). With seeds 1-3 (constant: queue 5; annealed: E-P, E-P2) this gives five
# seeds per condition and arm. Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.14 (python -B -m horizon.readings ep4).
# DEV split; NOT the frozen cycle. One CPU process, BelowNormal, idempotent. Constant runs first (cheap), then the annealed ones.
#
# Usage: nohup bash scripts/run_ep4.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

CUR=1,2,3,4,6,8,12,16
job() {   # arm seed decay-fraction (0 = constant lr)
  local arm="$1" seed="$2" frac="$3" id suffix="" extra=""
  if [ "$frac" != "0" ]; then suffix="-lrdecay0.3"; extra="--lr-decay-fraction $frac"; fi
  id="perm4-$arm-w64-s$seed-cur$suffix"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (CPU, 1 parcacik, 15000 adim, lr 0.003, mufredat $CUR, lr azaltma kesri $frac)"
  $PY -B -m horizon.depth_run --task perm4 --arm "$arm" --width 64 --seed "$seed" --steps 15000 --lr 0.003 --curriculum "$CUR" $extra --out "$OUT/$id.json" --log-every 2000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; fi
}

for seed in 4 5; do for arm in gru lstm; do job "$arm" "$seed" 0; done; done
for seed in 4 5; do for arm in gru lstm; do job "$arm" "$seed" 0.3; done; done
log "Z1 kuyruk EP4 BITTI"
bash scripts/run_ep5.sh      # the parameter-matched annealed GRU w128 / LSTM w112 and their long evaluation (note 5.17); added at 08:20, before this script started
