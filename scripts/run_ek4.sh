#!/usr/bin/env bash
# E-K4: does the S4 Core's margin erode when its width (state dimension per class) shrinks? S4 Core at widths 32 (state dim / classes = 1.33) and 48 (2.0), seed 1, EXACTLY the recipe of the
# stored baseline perm4-core-w64-s1-cur (2.67; 15,000 updates, lr 0.003 constant, curriculum 1..16), then the margin profile (note 5.20 tool, `erosion`) of the three. Note: w32 and w48 have one
# attention head (head dimension 32 / 48), the w64 baseline two. Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.23
# (python -B -m horizon.readings ek4). DEV split; NOT the frozen cycle. One CPU process at a time (about 25-30 minutes per run), BelowNormal, idempotent, lock directory per job.
#
# Usage: nohup bash scripts/run_ek4.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
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
job() {   # width
  local id="perm4-core-w$1-s1-cur"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA Z1 $id (E-K4; CPU, 1 parcacik, 15000 adim, lr 0.003 sabit, genislik $1, mufredat $CUR)"
  $PY -B -m horizon.depth_run --task perm4 --arm core --width "$1" --seed 1 --steps 15000 --lr 0.003 --curriculum "$CUR" --out "$OUT/$id.json" --log-every 1000 >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id: sonuc dosyasi yok (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

job 32
job 48

MP="$OUT/margin-profile-s4core-width"
if [ -f "$OUT/perm4-core-w32-s1-cur.pt" ] && [ -f "$OUT/perm4-core-w48-s1-cur.pt" ] && [ ! -f "$MP.json" ]; then
  log "BASLA E-K4 marj profili (CPU, 1 parcacik): S4 Core w32 / w48 / w64, 1.024 konum, n = 100, egitimsiz"
  $PY -B -m horizon.margin_profile "$OUT/perm4-core-w32-s1-cur.pt" "$OUT/perm4-core-w48-s1-cur.pt" "$OUT/perm4-core-w64-s1-cur.pt" --out "$MP.json" > "$MP.txt" 2>&1
  if [ -f "$MP.json" ]; then log "BITTI E-K4 marj profili (bkz $MP.txt)"; else log "HATA E-K4 marj profili: sonuc dosyasi yok (bkz $MP.txt)"; fi
fi
log "E-K4 kuyruk BITTI"
