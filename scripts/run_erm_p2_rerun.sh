#!/usr/bin/env bash
# A4 (god 2026-10-16): B_orc "muafiyet" kosusunun TEKRARI — kok-neden teshisi sonrasi.
# 15-Eki 05:01 kosusu +30:17'de DURDURULDU (yavas; vektorize oncesi ArmOrc) ve
# stdout blok-tamponu yuzunden ilerleme gorunmezdi. Bu betik:
#   - birebir ayni komutlari kosar (train/eval/g03; kural degisikligi YOK),
#   - PYTHONUNBUFFERED=1 ile satir-satir loglar (yalnizca loglama; matematik ayni),
#   - dis zaman asimi YOK; BelowNormal; atomik kilit ayni sema.
# Muafiyet: P1!=KALDI kosuluna ragmen P2 (kehanet-yazma) — god onayi 2026-10-15T01:59:15Z
# (ERM_PROB_ZEYL_TASLAK.md Z4); A4 yeniden-kosu talimati 2026-10-16 (god, rlt-ops).
# Yetki: A4 siniri 10 CPU-h. Kural dosyalarina dokunmaz.
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
OUT=results/ERM-S0
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

ID=erm-B_orc-s1
if [ ! -f "$OUT/$ID.json" ]; then
  if mkdir "$OUT/$ID.lock" 2>/dev/null; then
    echo "--- start $(date '+%Y-%m-%d %H:%M:%S') (muafiyet TEKRAR; vektorize; unbuffered)" >> "$OUT/$ID.log"
    log "BASLA ERM $ID (prob P2 muafiyet TEKRAR, vektorize, CPU, 1 parcacik, 6000 adim)"
    $PY -B -m horizon.erm_run train --arm B_orc --seed 1 --steps 6000 --batch 32 --lr 0.003 --log-every 1000 --out "$OUT/$ID.json" >> "$OUT/$ID.log" 2>&1
    if [ -f "$OUT/$ID.json" ]; then log "BITTI ERM $ID"; else log "HATA ERM $ID (bkz $OUT/$ID.log)"; rmdir "$OUT/$ID.lock" 2>/dev/null; fi
  else
    log "ATLA ERM $ID (kilitli)"
  fi
else
  log "ATLA ERM $ID (json zaten var)"
fi
if [ -f "$OUT/$ID.json" ] && [ ! -f "$OUT/$ID-reading.json" ]; then
  if mkdir "$OUT/$ID-eval.lock" 2>/dev/null; then
    log "BASLA ERM $ID degerlendirme (CPU, n=200/hucre)"
    $PY -B -m horizon.erm_run eval --weights "$OUT/$ID.pt" --d 1,2,3 --n 200 --out "$OUT/$ID-reading.json" >> "$OUT/$ID-eval.log" 2>&1
    if [ -f "$OUT/$ID-reading.json" ]; then log "BITTI ERM $ID degerlendirme"; else log "HATA ERM $ID degerlendirme (bkz $OUT/$ID-eval.log)"; rmdir "$OUT/$ID-eval.lock" 2>/dev/null; fi
  else
    log "ATLA ERM $ID degerlendirme (kilitli)"
  fi
fi
if [ -f "$OUT/$ID-reading.json" ] && [ ! -f "$OUT/erm-P2.json" ]; then
  if mkdir "$OUT/erm-P2.lock" 2>/dev/null; then
    $PY -B -m horizon.erm_run g03 --kural p2 --b "$OUT/$ID-reading.json" --a "$OUT/erm-A-s1-reading.json" --out "$OUT/erm-P2.json" >> "$OUT/erm-P2.log" 2>&1
    log "ERM P2 hukum: $($PY -B -c 'import json; print(json.load(open("results/ERM-S0/erm-P2.json"))["reading"]["verdict"])' 2>/dev/null || echo BILINMIYOR)"
  fi
fi
log "ERM prob-P2 kuyruk BITTI (muafiyet TEKRAR)"
