#!/usr/bin/env bash
# E-A: why does the S5 Core break at 4-8x the trained length? Inference-time attention interventions on the SAVED weights (no training, dev split).
# Pre-registered reading rule and predictions: notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5 (committed before this ran).
# One CPU process, BelowNormal; idempotent (skips when the result file exists).
#
# Usage: nohup bash scripts/run_ea.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ -f "$OUT/perm5-core-intervention.json" ]; then exit 0; fi
log "BASLA E-A (CPU, 1 parcacik): S5 Core dikkat mudahaleleri, egitimsiz, tohum 1-3"
$PY -B -m horizon.attention_intervention "$OUT/perm5-core-w128-s1-cur.pt" --seeds 1,2,3 --out "$OUT/perm5-core-intervention.json" > "$OUT/perm5-core-intervention.txt" 2>&1
if [ -f "$OUT/perm5-core-intervention.json" ]; then log "BITTI E-A (bkz $OUT/perm5-core-intervention.txt)"; else log "HATA E-A: sonuc dosyasi yok (bkz $OUT/perm5-core-intervention.txt)"; fi
