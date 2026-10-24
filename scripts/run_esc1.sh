#!/usr/bin/env bash
# E-SC1 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.74): can the RLT state, as the only channel between windows, carry
# a fact one window back? mline.carry via mline.rltm_train --arm C: rows of two 512-byte windows, every fact one window back, the
# cross-window answers weighted 0.5, FineWeb-Edu (user rule, section 5.70), L1's weights (trained on CNN; recorded), seed 1, 2,000
# updates, a checkpoint every 100. Waits for E-RM1W's queue and a free GPU; a 10-update smoke test first (timing only).
# scripts/pause_for_user.ps1 kills it; a relaunch resumes from latest.pt.
# Usage: nohup bash scripts/run_esc1.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. MLINE_DATA=__RLT_ROOT__/mline-data/fineweb-edu-bytes
OUT=results/ESC1
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.74"
INIT=results/L1-scratch-w928-s401/latest.pt
ID=C-d1-w50-s1
ARGS="--arm C --init $INIT --seed 1 --steps 2000 --windows 2 --phases d1 --answer-weight 0.5 --eval-d 1,2 --final-d 1,2,3 --ckpt-every 100"
log() { echo "[$(date +%H:%M:%S)] [esc1] $*" >> "$LOG"; }
busy() {   # long GPU runs (a venv run shows twice: the launcher and its interpreter); an empty answer counts as none
  local n
  n=$(powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'mline\.(longrun|rltm_train|rl0)' }).Count" 2>/dev/null | tr -d '\r ')
  echo "${n:-0}"
}

if [ -f "$OUT/$ID/final.json" ]; then log "E-SC1 $ID zaten bitmis"; else
  log "E-SC1 bekliyor: E-RM1W kuyrugu bitince ve GPU bosalinca"
  until grep -q "\[erm1w\] E-RM1W kuyrugu BITTI" "$LOG"; do sleep 120; done
  until [ "$(busy)" = "0" ]; do sleep 60; done
  mkdir -p "$OUT"
  if [ ! -f "$OUT/smoke-C/smoke.json" ]; then
    $PY -B -m mline.rltm_train $ARGS --out "$OUT/smoke-C" --smoke 10 >> "$OUT/smoke-C.log" 2>&1 || log "HATA E-SC1 duman testi (bkz $OUT/smoke-C.log)"
    log "E-SC1 duman testi: $("$PY" -B -c "import json; r = json.load(open('$OUT/smoke-C/smoke.json', encoding='utf8')); print('%.2f s/guncelleme, en yuksek ayrilmis %.2f GiB' % (r['seconds_per_update'], r['peak_reserved_gib']))" 2>/dev/null)"
  fi
  if [ ! -f "$OUT/$ID/latest.pt" ]; then
    "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
      "E-SC1-$ID" ESC1 claude-4797b78c "MLINE_DATA=fineweb-edu-bytes python -B -m mline.rltm_train $ARGS --out $OUT/$ID" "$P" '{"seed": 1}' "$0" >> "$LOG" 2>&1 \
      || log "MANIFEST HATA E-SC1"
    log "BASLA E-SC1 $ID (C: durum pencereler arasi tek kanal, 2 pencere, d = 1, cevap agirligi 0,5, FineWeb-Edu, 2000 guncelleme)"
  else
    log "SURDUR E-SC1 $ID (latest.pt'den)"
  fi
  $PY -B -m mline.rltm_train $ARGS --out "$OUT/$ID" >> "$OUT/$ID.log" 2>&1
  if [ -f "$OUT/$ID/final.json" ]; then log "BITTI E-SC1 $ID"; else log "HATA E-SC1 $ID (bkz $OUT/$ID.log)"; fi
fi
if [ -f "$OUT/$ID/final.json" ]; then
  log "E-SC1 okuma: $("$PY" -B -m mline.rltm_train --reading-carry "$OUT/$ID/final.json" 2>/dev/null | "$PY" -B -c "import json, sys; r = json.load(sys.stdin); print(r['label'], 'first_d1', round(r['first_d1'], 3), 'exact_d1', round(r['exact_d1'], 3), 'off', round(r['first_d1_off'], 3), round(r['exact_d1_off'], 3))" 2>/dev/null)"
fi
log "E-SC1 kuyrugu BITTI"
