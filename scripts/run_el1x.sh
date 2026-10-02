#!/usr/bin/env bash
# E-L1X: does the L1 byte model keep working on documents far longer than the 512 bytes it was trained on, and if not, is it the attention memory or the carried state? Evaluation only,
# CPU, on explicit L1 snapshots (default: the 6,000-update one). Reading rule written before this ran: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.35 (the reading is printed by
# mline.long_doc). Per snapshot: the registered run (32 articles x 4,096 bytes; the verdict) and the descriptive 8,192-byte run (16 articles). One CPU process, BelowNormal (the L1 training
# run owns the GPU and is not touched); idempotent.
#
# Usage: nohup bash scripts/run_el1x.sh [snapshot ...] > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=results/L1-scratch-w928-s401
OUT=results/L1-long-doc
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
mkdir -p "$OUT"

if [ "$#" -gt 0 ]; then SNAPS=("$@"); else SNAPS=("$RUN/snapshots/step006000.pt"); fi
for snap in "${SNAPS[@]}"; do
  name=$(basename "$snap" .pt)
  if [ ! -f "$snap" ]; then continue; fi
  if [ ! -f "$OUT/$name-long-doc.json" ]; then
    log "BASLA E-L1X ($name; CPU, 1 parcacik): 4.096 bayt, 32 makale: kesintisiz kosu / pencereli bellek / durum sifirlama / bagimsiz parcalar, egitimsiz"
    $PY -B -m mline.long_doc "$snap" --out "$OUT" > "$OUT/$name-long-doc.txt" 2>&1
    if [ -f "$OUT/$name-long-doc.json" ]; then log "BITTI E-L1X $name (bkz $OUT/$name-long-doc.txt)"; else log "HATA E-L1X $name: sonuc dosyasi yok (bkz $OUT/$name-long-doc.txt)"; fi
  fi
  if [ ! -f "$OUT/$name-long-doc-8192.json" ]; then
    log "BASLA E-L1X ($name; betimleyici 8.192 bayt, 16 makale)"
    $PY -B -m mline.long_doc "$snap" --length 8192 --n 16 --out "$OUT" > "$OUT/$name-long-doc-8192.txt" 2>&1
    if [ -f "$OUT/$name-long-doc-8192.json" ]; then log "BITTI E-L1X $name 8192 (bkz $OUT/$name-long-doc-8192.txt)"; else log "HATA E-L1X $name 8192: sonuc dosyasi yok (bkz $OUT/$name-long-doc-8192.txt)"; fi
  fi
done
log "E-L1X kuyruk BITTI"
