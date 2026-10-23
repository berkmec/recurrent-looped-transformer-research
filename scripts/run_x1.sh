#!/usr/bin/env bash
# E-X1 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.59): the standard transformer baseline (mline.tfm, arm tfm6: 6 layers,
# width 528, 8 heads, 20.37M parameters) on L1's recipe, data, seed and schedule (30,518 updates). Waits until E-RM stage 1 has ended
# (any final line of its queue; user priority 2026-10-23 13:0x moved stage 1 ahead) and no other long run is on the GPU, logs the VRAM
# three minutes in, writes the reading.
# Usage: nohup bash scripts/run_x1.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
OUT=results/L1-tfm6-w528-s401
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.59"
CMD="python -B -m mline.longrun --out $OUT --recipe m2-draft --tokens 500000000 --arm tfm6 --width 528 --heads 8"
log() { echo "[$(date +%H:%M:%S)] [x1] $*" >> "$LOG"; }
busy() {   # long-run processes (a venv run shows twice: the launcher and its interpreter); an empty answer counts as none, the E-L1C line is the primary gate
  local n
  n=$(powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'mline\.(longrun|rltm_train)' }).Count" 2>/dev/null | tr -d '\r ')
  echo "${n:-0}"
}

if [ -f "$OUT/complete.json" ]; then log "ATLA E-X1 (complete.json var)"; exit 0; fi
log "E-X1 bekliyor: E-RM Asama 1 kuyrugu bitince (ya da durunca) ve GPU'da baska uzun kosu kalmayinca baslar"
until grep -q -E "\[erm1\] (E-RM Asama 1 kuyrugu BITTI|Asama 1 DURDU|HATA Asama 1)" "$LOG"; do sleep 120; done
until [ "$(busy)" = "0" ]; do sleep 60; done
mkdir -p "$OUT"
"$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
  E-X1-L1-tfm6-w528-s401 L1-tfm6-w528-s401 claude-4797b78c "$CMD" "$P" '{"seed": 401}' "$0" >> "$LOG" 2>&1 || log "MANIFEST HATA E-X1"
log "BASLA E-X1 tfm6 w528 (GPU, 30.518 guncelleme, L1 tarifi)"
"$PY" -B -m mline.longrun --out "$OUT" --recipe m2-draft --tokens 500000000 --arm tfm6 --width 528 --heads 8 >> "$OUT/train.log" 2>&1 &
TP=$!
sleep 180
log "E-X1 VRAM 3 dk sonra: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | tr -d '\r')"
wait "$TP"
if [ -f "$OUT/complete.json" ]; then
  log "BITTI E-X1"
  "$PY" -B -m mline.x1_reading --out "$OUT/reading-x1.json" > /dev/null 2>&1 \
    && log "BITTI E-X1 okuma: $("$PY" -B -c "import json; r=json.load(open('$OUT/reading-x1.json')); print(r['verdict'], 'delta_end', r.get('delta_last'), '| vs A0:', r.get('transformer_vs_a0', {}).get('verdict'))" 2>/dev/null)" \
    || log "HATA E-X1 okuma"
else
  log "HATA E-X1 (bkz $OUT/train.log)"
fi
