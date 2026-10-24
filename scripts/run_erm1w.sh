#!/usr/bin/env bash
# E-RM1W (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.73): stage 1's B_ds run with one change, the cross-window answers
# weighted 0.5 in the loss (mline.rltm_train.objective; bug hunt 5.71.2: they were 0.2 % of it), on the FineWeb-Edu corpus (user rule,
# section 5.70: no training on the CNN corpus). Seed 1, 2,000 updates, a checkpoint every 100: scripts/pause_for_user.ps1 kills the
# process and a relaunch of this script resumes from latest.pt. The starting weights are L1's (trained on CNN; recorded in 5.73).
# Usage: nohup bash scripts/run_erm1w.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. MLINE_DATA=__RLT_ROOT__/mline-data/fineweb-edu-bytes
OUT=results/ERM1W
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.73"
INIT=results/L1-scratch-w928-s401/latest.pt
ID=B_ds-w50-s1
log() { echo "[$(date +%H:%M:%S)] [erm1w] $*" >> "$LOG"; }
busy() {   # long GPU runs (a venv run shows twice: the launcher and its interpreter); an empty answer counts as none
  local n
  n=$(powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'mline\.(longrun|rltm_train|rl0)' }).Count" 2>/dev/null | tr -d '\r ')
  echo "${n:-0}"
}

if [ -f "$OUT/$ID/final.json" ]; then log "E-RM1W $ID zaten bitmis"; else
  until [ "$(busy)" = "0" ]; do sleep 60; done
  mkdir -p "$OUT"
  if [ ! -f "$OUT/$ID/latest.pt" ]; then
    "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
      "E-RM1W-$ID" ERM1W claude-4797b78c "MLINE_DATA=fineweb-edu-bytes python -B -m mline.rltm_train --arm B_ds --init $INIT --seed 1 --steps 2000 --out $OUT/$ID --ut-chunk 64 --answer-weight 0.5 --ckpt-every 100" "$P" '{"seed": 1}' "$0" >> "$LOG" 2>&1 \
      || log "MANIFEST HATA E-RM1W"
    log "BASLA E-RM1W $ID (B_ds, cevap agirligi 0,5, FineWeb-Edu, 2000 guncelleme)"
  else
    log "SURDUR E-RM1W $ID (latest.pt'den)"
  fi
  "$PY" -B -m mline.rltm_train --arm B_ds --init "$INIT" --seed 1 --steps 2000 --out "$OUT/$ID" --ut-chunk 64 --answer-weight 0.5 \
    --ckpt-every 100 >> "$OUT/$ID.log" 2>&1
  if [ -f "$OUT/$ID/final.json" ]; then log "BITTI E-RM1W $ID"; else log "HATA E-RM1W $ID (bkz $OUT/$ID.log)"; fi
fi
if [ -f "$OUT/$ID/final.json" ]; then
  log "E-RM1W okuma: $("$PY" -B -m mline.rltm_train --reading-objective "$OUT/$ID/final.json" 2>/dev/null | "$PY" -B -c "import json, sys; r = json.load(sys.stdin); print(r['label'], 'first_d1', round(r['first_d1'], 3), 'exact_d2', round(r['exact_d2'], 3), 'read_off', round(r['exact_d2_read_off'], 3))" 2>/dev/null)"
fi
log "E-RM1W kuyrugu BITTI"
