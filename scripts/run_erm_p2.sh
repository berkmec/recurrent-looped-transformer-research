#!/usr/bin/env bash
# Prob P2: kehanet-yazma (B_orc). YALNIZ P1 KALDI ise koşar (ön-kayıt koşulu);
# P1 GECTI ise P2 iptal (~14 dk tasarruf). Ön-kayıt: notes/ERM_PROB_YAZMA_ONKAYIT.md.
# Yetki: T8 insan onayı + god marj onayı GEREKİR (bu betik ONAYSIZ ÇALIŞTIRILMAZ).
# Kural: acc-only (d=3 >= 0,95 VE A <= 0,30) -> GECTI (YAZMA_ÖLÜ adayı) / KALDI (MİMARİ_YETERSİZ adayı).
# Tek parçacık, atomik kilit, idempotent, BelowNormal, 1 iş parçacığı (~14 dk CPU).
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ERM-S0
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ -f "$OUT/erm-P1.json" ]; then
  P1V=$($PY -B -c 'import json; print(json.load(open("results/ERM-S0/erm-P1.json"))["reading"]["verdict"])' 2>/dev/null || echo BILINMIYOR)
else
  P1V="YOK"
fi
if [ "$P1V" != "KALDI" ]; then
  log "ERM P2 iptal/atlandi (P1 hukum: $P1V; P2 yalniz KALDI'da kosar)"
  exit 0
fi

ID=erm-B_orc-s1
if [ ! -f "$OUT/$ID.json" ]; then
  if mkdir "$OUT/$ID.lock" 2>/dev/null; then
    echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$ID.log"
    log "BASLA ERM $ID (prob P2 kehanet-yazma, CPU, 1 parcacik, 6000 adim, ~14 dk)"
    $PY -B -m horizon.erm_run train --arm B_orc --seed 1 --steps 6000 --batch 32 --lr 0.003 --log-every 1000 --out "$OUT/$ID.json" >> "$OUT/$ID.log" 2>&1
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
if [ -f "$OUT/$ID-reading.json" ] && [ ! -f "$OUT/erm-P2.json" ]; then
  if mkdir "$OUT/erm-P2.lock" 2>/dev/null; then
    $PY -B -m horizon.erm_run g03 --kural p2 --b "$OUT/$ID-reading.json" --a "$OUT/erm-A-s1-reading.json" --out "$OUT/erm-P2.json" >> "$OUT/erm-P2.log" 2>&1
    log "ERM P2 hukum: $($PY -B -c 'import json; print(json.load(open("results/ERM-S0/erm-P2.json"))["reading"]["verdict"])' 2>/dev/null || echo BILINMIYOR)"
  fi
fi
log "ERM prob-P2 kuyruk BITTI"
