#!/usr/bin/env bash
# E-RL1 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.69): E-RL0 with the paper's addition budget on the GPU (mline.rl0 --config rl1):
# L1 then A0F, then the registered reading (E-RL0's rule). Mode after-stage1 (default): waits for the stage-1 queue's final line ([erm1b]).
# Mode between: waits for stage 1's first run (B_ds seed 1) to finish, runs, then resumes stage 1 (scripts/run_erm_stage1_ut_resume.sh); the
# stage-1 runner must have been stopped before that run ended. Each arm first times a smoke run on unregistered cells. The E-X1 runner waits
# for this queue's final line as well as stage 1's.
# Usage: nohup bash scripts/run_rl1.sh [after-stage1|between] > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
MODE="${1:-after-stage1}"
OUT=results/RL1
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.69"
log() { echo "[$(date +%H:%M:%S)] [rl1] $*" >> "$LOG"; }
busy() {   # long GPU runs (a venv run shows twice: the launcher and its interpreter); an empty answer counts as none
  local n
  n=$(powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'mline\.(longrun|rltm_train)|mline\.rl0 .*--device cuda' }).Count" 2>/dev/null | tr -d '\r ')
  echo "${n:-0}"
}
timing() {  # smoke directory -> seconds per SFT update and per RL step, projected full run
  "$PY" -B -c "
import json, sys
def per(f):
    rows = [json.loads(l) for l in open(f, encoding='utf8') if l.strip()]
    return (rows[-1]['seconds'] - rows[0]['seconds']) / max(1, len(rows) - 1)
s, r = per(sys.argv[1] + '/sft.jsonl'), per(sys.argv[1] + '/rl.jsonl')
print('SFT %.2f s/guncelleme, RL %.2f s/adim -> egitim ~%.1f sa (degerlendirmeler haric)' % (s, r, (2000 * s + 500 * r) / 3600))" "$1" 2>/dev/null
}
mkdir -p "$OUT"

if [ "$MODE" = "between" ]; then
  log "E-RL1 bekliyor (araya): Asama 1'in ilk kosusu (B_ds tohum 1) bitince"
  until [ -f results/ERM-stage1-ut/B_ds-s1/final.json ]; do sleep 60; done
else
  log "E-RL1 bekliyor: Asama 1 kuyrugu bitince ([erm1b] son satiri)"
  until grep -q -E "\[erm1b\] (E-RM Asama 1 kuyrugu BITTI|Asama 1 DURDU|HATA Asama 1)" "$LOG"; do sleep 120; done
fi
until [ "$(busy)" = "0" ]; do sleep 60; done
"$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
  E-RL1-L1-A0F RL1 claude-4797b78c "python -B -m mline.rl0 run <L1 | A0F latest.pt> --out $OUT/<arm> --seed 1 --config rl1 --device cuda; python -B -m mline.rl0 reading" "$P" '{"seed": 1}' "$0" >> "$LOG" 2>&1 \
  || log "MANIFEST HATA E-RL1"

arm() {  # arm name, weights
  local name="$1" weights="$2"
  [ -f "$OUT/$name/final.json" ] && return 0
  if [ ! -f "$OUT/smoke-$name/final.json" ]; then
    "$PY" -B -m mline.rl0 run "$weights" --out "$OUT/smoke-$name" --seed 1 --config rl1 --device cuda --threads 2 --smoke > "$OUT/smoke-$name.txt" 2>&1
    log "E-RL1 $name duman testi (kayit disi hucreler): $(timing "$OUT/smoke-$name"); VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d '\r')"
  fi
  log "BASLA E-RL1 $name (GPU; SFT 2.000 x 512 problem + RL 500 x 32 x 8)"
  "$PY" -B -m mline.rl0 run "$weights" --out "$OUT/$name" --seed 1 --config rl1 --device cuda --threads 2 >> "$OUT/$name.txt" 2>&1
  if [ -f "$OUT/$name/final.json" ]; then log "BITTI E-RL1 $name"; else log "HATA E-RL1 $name (bkz $OUT/$name.txt)"; fi
}

arm L1 results/L1-scratch-w928-s401/latest.pt
arm A0F results/L1-a0-w928-s401/latest.pt
if [ -f "$OUT/L1/final.json" ] && [ -f "$OUT/A0F/final.json" ]; then
  "$PY" -B -m mline.rl0 reading "$OUT/L1/final.json" "$OUT/A0F/final.json" > "$OUT/reading.json" 2> "$OUT/reading.err" \
    && log "BITTI E-RL1 okuma: $("$PY" -B -c "import json; r = json.load(open('$OUT/reading.json', encoding='utf8')); print('S1 L1', r['q1']['L1']['verdict'], 'delta %+.3f' % r['q1']['L1']['delta'], '| S2', r['q2']['verdict'], 'D %+.3f' % r['q2']['delta'], '| SFT LEARNED', r['sft_learned'])" 2>/dev/null)" \
    || log "HATA E-RL1 okuma (bkz $OUT/reading.err)"
fi
log "E-RL1 kuyruk BITTI"
if [ "$MODE" = "between" ]; then
  log "Asama 1 surduruluyor (E-RL1 araya girmisti)"
  nohup bash scripts/run_erm_stage1_ut_resume.sh > /dev/null 2>&1 &
fi
