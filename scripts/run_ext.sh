#!/usr/bin/env bash
# Descriptive extension: the E-A5 reading (attention windows + oracle state re-injection) and the E-A9 reading (self-snap, 1,024 positions, no long part) on later Core models of the S5 task
# (E-D5 sigma 0.1, E-K5 w256 / w192, E-F5 variants, E-R5 seeds, E-A10 core_clean ...): they are DESCRIPTIVE and never enter a registered verdict. One CPU thread, BelowNormal, idempotent
# (a model is skipped when its result file exists). Roughly 10 minutes per model.
#
# Usage: nohup bash scripts/run_ext.sh perm5-core-w128-s1-cur-lrdecay0.3-noise0.1 [more model basenames under results/Z1-pilot, without .pt] > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Z=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

for m in "$@"; do
  if [ ! -f "$Z/$m.pt" ]; then log "ATLANDI ext $m: model dosyasi yok"; continue; fi
  if [ ! -f "$Z/ext-$m-erosion.json" ]; then
    log "BASLA ext $m (E-A5 okumasi: dikkat pencereleri + kahin enjeksiyon; betimleyici)"
    $PY -B -m horizon.erosion_source "$Z/$m.pt" --out "$Z/ext-$m-erosion.json" > "$Z/ext-$m-erosion.txt" 2>&1
    if [ -f "$Z/ext-$m-erosion.json" ]; then log "BITTI ext $m erosion"; else log "HATA ext $m erosion (bkz $Z/ext-$m-erosion.txt)"; fi
  fi
  if [ ! -f "$Z/ext-$m-snap.json" ]; then
    log "BASLA ext $m (E-A9 okumasi: kendi-sinifina yapisma, 1.024 konum; betimleyici)"
    $PY -B -m horizon.self_snap "$Z/$m.pt" --no-long --out "$Z/ext-$m-snap.json" > "$Z/ext-$m-snap.txt" 2>&1
    if [ -f "$Z/ext-$m-snap.json" ]; then log "BITTI ext $m snap"; else log "HATA ext $m snap (bkz $Z/ext-$m-snap.txt)"; fi
  fi
done
log "ext kuyruk BITTI ($*)"
