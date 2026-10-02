#!/usr/bin/env bash
# E-A10 (program note 5.28): does the S5 Core with a gated Hopfield cleanup in the feedback path (`core_clean`, horizon/cleanup.py), trained with carried-state noise, lose the long-position
# break the plain Core has (E-A5: the deviation of the carried state from its class grows with the state's age)? S5 `core_clean` w128, seed 1, 30,000 updates, lr 0.003 annealed to ~0 over the last
# 30 % (exactly the E-P2 / E-D5 recipe), cleanup gate initialised at tanh(0.3). Runs: (a) state noise sigma 0.1 [the primary run], (b) no noise, (c) sigma 0.2. `a` runs only (a); `all` (the
# default) runs (a), (b), (c). One CPU process at a time (about 75-90 minutes per run), BelowNormal, idempotent; an atomic lock directory per job lets two queue chains share it.
# Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.28 (python -B -m horizon.readings ea10). DEV split; NOT the frozen cycle.
#
# Usage: nohup bash scripts/run_ea10.sh [a|all] > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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
job() {   # id, state-noise sigma (0 = none)
  local id="$1" sigma="$2"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (E-A10; CPU, 1 parcacik, 30000 adim, lr 0.003 son %30'da ~0'a, core_clean (kapali Hopfield temizleme, kapi tanh(0.3)), durum gurultusu sigma $sigma, mufredat $CUR)"
  $PY -B -m horizon.depth_run --task perm5 --arm core_clean --width 128 --seed 1 --steps 30000 --lr 0.003 --curriculum "$CUR" --lr-decay-fraction 0.3 --state-noise "$sigma" --cleanup-open 0.3 \
    --out "$OUT/$id.json" --log-every 2000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

job perm5-core_clean-w128-s1-cur-lrdecay0.3-noise0.1-open0.3 0.1
if [ "${1:-all}" != "a" ]; then
  job perm5-core_clean-w128-s1-cur-lrdecay0.3-open0.3 0
  job perm5-core_clean-w128-s1-cur-lrdecay0.3-noise0.2-open0.3 0.2
fi
log "E-A10 kuyruk BITTI (${1:-all})"
