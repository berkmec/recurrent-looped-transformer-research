#!/usr/bin/env bash
# E-RM yedek merdiven 3. basamak: B_rmt (RMT-sadık, kapısız yuva-belirteci taşıma).
# Tetikler: G0.3 KALDI + B_comp KALDI; B_32 atlandı (god 2026-10-14T17-58-18-939Z-73f556).
# Yetki: ayni agree (T7 kapsami, ~25 dk CPU). Kural: B_rmt-özel acc-only
# (`horizon.erm_run.reading_rmt`: d=3 >= 0,95 VE A <= 0,30 -> GECTI; telemetri betimleyici).
# KALDI -> üç yedek tükendi: E-RM DURUR, kayıt düşer (gözetimli-yazma probu T8'e girer).
# Tek parçacık, atomik kilit, idempotent, BelowNormal, 1 iş parçacığı.
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ERM-S0
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

ID=erm-B_rmt-s1
if [ ! -f "$OUT/$ID.json" ]; then
  if mkdir "$OUT/$ID.lock" 2>/dev/null; then
    echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$ID.log"
    log "BASLA ERM $ID (merdiven-3 B_rmt, CPU, 1 parcacik, 6000 adim, ~25 dk)"
    $PY -B -m horizon.erm_run train --arm B_rmt --seed 1 --steps 6000 --batch 32 --lr 0.003 --log-every 1000 --out "$OUT/$ID.json" >> "$OUT/$ID.log" 2>&1
    if [ -f "$OUT/$ID.json" ]; then log "BITTI ERM $ID"; else log "HATA ERM $ID (bkz $OUT/$ID.log)"; rmdir "$OUT/$ID.lock" 2>/dev/null; fi
  fi
fi
if [ -f "$OUT/$ID.json" ] && [ ! -f "$OUT/$ID-reading.json" ]; then
  if mkdir "$OUT/$ID-eval.lock" 2>/dev/null; then
    log "BASLA ERM $ID degerlendirme (CPU, n=200/hucre)"
    $PY -B -m horizon.erm_run eval --weights "$OUT/$ID.pt" --d 1,2,3 --n 200 --out "$OUT/$ID-reading.json" >> "$OUT/$ID-eval.log" 2>&1
    if [ -f "$OUT/$ID-reading.json" ]; then log "BITTI ERM $ID degerlendirme"; else log "HATA ERM $ID degerlendirme (bkz $OUT/$ID-eval.log)"; rmdir "$OUT/$ID-eval.lock" 2>/dev/null; fi
  fi
fi
if [ -f "$OUT/$ID-reading.json" ] && [ ! -f "$OUT/erm-Brmt-G03.json" ]; then
  if mkdir "$OUT/erm-Brmt-G03.lock" 2>/dev/null; then
    $PY -B -m horizon.erm_run g03 --kural rmt --b "$OUT/$ID-reading.json" --a "$OUT/erm-A-s1-reading.json" --out "$OUT/erm-Brmt-G03.json" >> "$OUT/erm-Brmt-G03.log" 2>&1
    log "ERM B_rmt hukum: $($PY -B -c 'import json; print(json.load(open("results/ERM-S0/erm-Brmt-G03.json"))["reading"]["verdict"])' 2>/dev/null || echo BILINMIYOR)"
  fi
fi
log "ERM merdiven-3 kuyruk BITTI"
