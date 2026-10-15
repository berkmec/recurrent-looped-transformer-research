#!/usr/bin/env bash
# Prob P1b: yardımlı-yazma, aux 0.3 (P1 aux-1.0 DIVERGE sonrası stabilize tekrar).
# Ön-kayıt: notes/ERM_PROB_YAZMA_ONKAYIT.md (P1) + god 2026-10-15 seçimi (i) minimal müdahale.
# Yetki: T8 insan onayı + god marj onayı GEREKİR (bu betik ONAYSIZ ÇALIŞTIRILMAZ).
# Fark P1'den: --aux-weight 0.3 + yeni kimlik (erm-B_aux03-s1; P1 ciktiari korunur).
# Kural: G0.3 eşikleri + yardım-UYGULANDI -> GECTI (SİNYAL_YOK adayı) / KALDI+uygulandı -> P2.
# Tek parçacık, atomik kilit, idempotent, BelowNormal, 1 iş parçacığı (~17 dk CPU).
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ERM-S0
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

ID=erm-B_aux03-s1
if [ ! -f "$OUT/$ID.json" ]; then
  if mkdir "$OUT/$ID.lock" 2>/dev/null; then
    echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$ID.log"
    log "BASLA ERM $ID (prob P1b aux-0.3, CPU, 1 parcacik, 6000 adim, ~17 dk)"
    $PY -B -m horizon.erm_run train --arm B_aux --seed 1 --steps 6000 --batch 32 --lr 0.003 --aux-weight 0.3 --log-every 1000 --out "$OUT/$ID.json" >> "$OUT/$ID.log" 2>&1
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
if [ -f "$OUT/$ID-reading.json" ] && [ ! -f "$OUT/erm-P1b.json" ]; then
  if mkdir "$OUT/erm-P1b.lock" 2>/dev/null; then
    $PY -B -m horizon.erm_run g03 --kural p1 --b "$OUT/$ID-reading.json" --a "$OUT/erm-A-s1-reading.json" --out "$OUT/erm-P1b.json" >> "$OUT/erm-P1b.log" 2>&1
    log "ERM P1b hukum: $($PY -B -c 'import json; print(json.load(open("results/ERM-S0/erm-P1b.json"))["reading"]["verdict"])' 2>/dev/null || echo BILINMIYOR)"
  fi
fi
log "ERM prob-P1b kuyruk BITTI"
