#!/usr/bin/env bash
# E-BLIMP (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.67): BLiMP (nyu-mll/blimp at revision 877fba0801ff, 67 x 1,000 minimal pairs)
# on L1 (feedback on) and A0F (feedback off), both at 30,518 updates, then the registered reading. Evaluation only, CPU, one process with two
# threads, BelowNormal (the GPU belongs to stage 1); idempotent. Started at 23:53 next to E-L1X-F, free memory fell to 2.4 GB (the watch limit is 2.5),
# so it was stopped within a minute and now waits for E-L1X-F's queue to end.
# Usage: nohup bash scripts/run_blimp.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
DATA=__RLT_ROOT__/datasets/nyu-mll__blimp
OUT=results/BLIMP
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.67"
log() { echo "[$(date +%H:%M:%S)] [blimp] $*" >> "$LOG"; }
mkdir -p "$OUT"

log "E-BLIMP bekliyor: E-L1X-F kuyrugu bitince baslar (bellek: ikisi birlikte bos RAM'i 2,5 GB'nin altina indirdi)"
until grep -q -E "\[l1xf\] E-L1X-F kuyruk BITTI" "$LOG"; do sleep 120; done
"$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
  E-BLIMP-L1-A0F BLIMP claude-4797b78c "python -B -m mline.blimp score <L1 | A0F latest.pt> --data $DATA --out $OUT/<name>-30518.json; python -B -m mline.blimp reading" "$P" '{"seed": 401}' "$0" >> "$LOG" 2>&1 \
  || log "MANIFEST HATA E-BLIMP"

one() {  # name, weights
  [ -f "$OUT/$1-30518.json" ] && return 0
  log "BASLA E-BLIMP $1 (67 x 1.000 cift; CPU, 2 is parcacigi)"
  "$PY" -B -m mline.blimp score "$2" --data "$DATA" --out "$OUT/$1-30518.json" --threads 2 > "$OUT/$1-30518.txt" 2>&1
  if [ -f "$OUT/$1-30518.json" ]; then log "BITTI E-BLIMP $1: $(tail -n 1 "$OUT/$1-30518.txt" | cut -c1-220)"; else log "HATA E-BLIMP $1 (bkz $OUT/$1-30518.txt)"; fi
}

one L1 results/L1-scratch-w928-s401/latest.pt
one A0F results/L1-a0-w928-s401/latest.pt
if [ -f "$OUT/L1-30518.json" ] && [ -f "$OUT/A0F-30518.json" ]; then
  "$PY" -B -m mline.blimp reading "$OUT/L1-30518.json" "$OUT/A0F-30518.json" > "$OUT/reading.json" 2> "$OUT/reading.err" \
    && log "BITTI E-BLIMP okuma: $("$PY" -B -c "import json; r = json.load(open('$OUT/reading.json', encoding='utf8')); print(r['verdict'], 'delta %+.2f CI [%.2f, %.2f] L1 %.2f A0F %.2f' % (r['delta'], r['ci'][0], r['ci'][1], r['l1'], r['a0']))" 2>/dev/null)" \
    || log "HATA E-BLIMP okuma (bkz $OUT/reading.err)"
fi
log "E-BLIMP kuyruk BITTI"
