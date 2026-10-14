#!/usr/bin/env bash
# E-RM Aşama 0 kapıları (R-5, kullanıcı onaylı adım 2; ön-kayıt: notes/ERM_S0_ONKAYIT_2026-10-14.md).
# Kayıtlı G0.3 kuralı: B d=3 acc>=0,95 VE A<=0,30 VE sign_ok VE yuva>=2 -> GECTI;
# A>0,50 -> KONTROL_BOZUK; B<0,95 -> KALDI (yedek merdiven SIRAYLA, bu betikte YOK).
# E-G kuyruğuyla PARALEL tek parçacık: atomik kilit dizini + idempotent (biten atlanır),
# BelowNormal, 1 iş parçacığı. G0.4 (GPU) bu betikte YOK — kapılar + ayrı god izni gerekir.
#
# Usage: nohup bash scripts/run_erm_s0.sh > /dev/null 2>&1 &   (izleme: ortak kuyruk günlüğü)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ERM-S0
LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

gate() {  # hizli kapi (g01|g02|g05)
  local id="erm-$1"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  log "BASLA ERM $id (CPU, hizli kapi)"
  $PY -B -m horizon.erm_run "$1" --out "$OUT/$id.json" >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI ERM $id"; else log "HATA ERM $id (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

trainjob() {  # kol tohum adim
  local id="erm-$1-s$2"
  if [ -f "$OUT/$id.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then return 0; fi
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/$id.log"
  log "BASLA ERM $id (CPU, 1 parcacik, $3 adim, batch 32, lr 0.003, ~22 dk)"
  $PY -B -m horizon.erm_run train --arm "$1" --seed "$2" --steps "$3" --batch 32 --lr 0.003 --log-every 1000 --out "$OUT/$id.json" >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI ERM $id"; else log "HATA ERM $id (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; fi
}

evaljob() {  # agirlik dosyasi koku (orn erm-B-s1)
  local id="$1"
  if [ -f "$OUT/$id-reading.json" ]; then return 0; fi
  if ! mkdir "$OUT/$id-eval.lock" 2>/dev/null; then return 0; fi
  log "BASLA ERM $id degerlendirme (CPU, n=200/hucre)"
  $PY -B -m horizon.erm_run eval --weights "$OUT/$id.pt" --d 1,2,3 --n 200 --out "$OUT/$id-reading.json" >> "$OUT/$id-eval.log" 2>&1
  if [ -f "$OUT/$id-reading.json" ]; then log "BITTI ERM $id degerlendirme"; else log "HATA ERM $id degerlendirme (bkz $OUT/$id-eval.log)"; rmdir "$OUT/$id-eval.lock" 2>/dev/null; fi
}

gate g01
gate g02
gate g05
trainjob B 1 6000
trainjob A 1 6000
trainjob Apr 1 6000
evaljob erm-B-s1
evaljob erm-A-s1
evaljob erm-Apr-s1
if [ -f "$OUT/erm-B-s1-reading.json" ] && [ -f "$OUT/erm-A-s1-reading.json" ] && [ ! -f "$OUT/erm-G03.json" ]; then
  if ! mkdir "$OUT/erm-G03.lock" 2>/dev/null; then :; else
    $PY -B -m horizon.erm_run g03 --b "$OUT/erm-B-s1-reading.json" --a "$OUT/erm-A-s1-reading.json" --out "$OUT/erm-G03.json" >> "$OUT/erm-G03.log" 2>&1
    log "ERM G0.3 hukum: $(cat $OUT/erm-G03.json | $PY -B -c 'import json,sys; print(json.load(sys.stdin)["reading"]["verdict"])')"
  fi
fi
log "ERM S0 kuyruk BITTI (G0.4 GPU haric; yedek merdiven KALDI hukumunde ayrica)"
