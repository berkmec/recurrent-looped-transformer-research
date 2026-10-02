#!/usr/bin/env bash
# E-P2 descriptive extras (program note 5.11, "betimleyici ek, karara girmez"): the E-B2 kick growth and class geometry (R/M) on the NEW lr-annealed S5 Core w128 weights, to be read next
# to the numbers the stored constant-lr S5 Core gave (results/Z1-pilot/state-growth.json, note 5.4.1). Evaluation only, same kick directions (seed 11) and settings as E-B2.
# One CPU process, BelowNormal, idempotent (skips when the result file exists or the weights do not).
#
# Usage: nohup bash scripts/run_ep2x.sh > /dev/null 2>&1 &        (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=results/Z1-pilot/perm5-core-w128-s1-cur-lrdecay0.3.pt
OUT=results/Z1-pilot/state-growth-s5core-lrdecay0.3
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ -f "$W" ] && [ ! -f "$OUT.json" ]; then
  log "BASLA E-P2x (CPU, 1 parcacik): E-B2 tekme buyumesi ve sinif geometrisi, lr azaltmali S5 Core w128, egitimsiz"
  $PY -B -m horizon.state_growth "$W" --out "$OUT.json" > "$OUT.txt" 2>&1
  if [ -f "$OUT.json" ]; then log "BITTI E-P2x (bkz $OUT.txt)"; else log "HATA E-P2x: sonuc dosyasi yok (bkz $OUT.txt)"; fi
fi

# E-M5 (note 5.20, descriptive): the logit-margin profile of the saved arms (survivors only), Core arms first. Models that exist at the time this runs.
Z=results/Z1-pilot
MP="$Z/margin-profile"
if [ ! -f "$MP.json" ]; then
  MODELS=()
  for m in perm5-core-w128-s1-cur perm5-core-w128-s1-cur-lrdecay0.3 perm4-core-w64-s1-cur perm4-gru-w64-s1-cur perm4-gru-w64-s1-cur-lrdecay0.3 \
           perm5-gru-w256-s1-cur perm5-gru-w256-s1-cur-lrdecay0.3 perm5-lstm-w208-s1-cur perm5-lstm-w208-s1-cur-lrdecay0.3; do
    if [ -f "$Z/$m.pt" ]; then MODELS+=("$Z/$m.pt"); fi
  done
  log "BASLA E-M5 (CPU, 1 parcacik): marj profili, ${#MODELS[@]} kayitli model, 1.024 konum, n = 100, egitimsiz"
  $PY -B -m horizon.margin_profile "${MODELS[@]}" --out "$MP.json" > "$MP.txt" 2>&1
  if [ -f "$MP.json" ]; then log "BITTI E-M5 (bkz $MP.txt)"; else log "HATA E-M5: sonuc dosyasi yok (bkz $MP.txt)"; fi
fi
