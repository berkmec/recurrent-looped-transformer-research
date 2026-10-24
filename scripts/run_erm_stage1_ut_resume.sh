#!/usr/bin/env bash
# Resume E-RM stage 1 with the parallel write (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.61.1) after a pause for the user: the
# runs, rule and extension rule of scripts/run_erm_stage1_ut.sh, but the gate is read from the G0.4b smoke results already measured
# (results/ERM-stage1-ut/smoke-*/smoke.json) instead of being run again: re-running a smoke test would resume its 20-update checkpoint and train
# on to 4,000 updates. Finished runs are skipped, an interrupted run resumes from its own checkpoint (mline.rltm_train). Same tag [erm1b], so the
# E-X1 runner still waits for this queue's final line.
# Usage: nohup bash scripts/run_erm_stage1_ut_resume.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
OUT=results/ERM-stage1-ut
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.61.1"
INIT=results/L1-scratch-w928-s401/latest.pt
UT=64
log() { echo "[$(date +%H:%M:%S)] [erm1b] $*" >> "$LOG"; }
busy() {   # long GPU runs (a venv run shows twice: the launcher and its interpreter); an empty answer counts as none
  local n
  n=$(powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'mline\.(longrun|rltm_train)' }).Count" 2>/dev/null | tr -d '\r ')
  echo "${n:-0}"
}
d2() {     # median exact accuracy at d = 2 (held-out real words) over the B_ds runs at update $1
  "$PY" -B -c "
import json, sys, statistics
vals = []
for s in (1, 2):
    rows = [json.loads(l) for l in open('$OUT/B_ds-s%d/probe.jsonl' % s, encoding='utf8') if l.strip()]
    vals += [r['eval']['naw']['natural']['2']['exact'] for r in rows if r['step'] == int(sys.argv[1])]
print(statistics.median(vals) if len(vals) == 2 else 'NA')" "$1" 2>/dev/null
}
gate() {   # smoke directory of A, smoke directory of B_ds -> 'PASS ...' or 'FAIL ...' (nothing if a smoke.json is missing)
  "$PY" -B -c "
import json, sys
a, b = (json.load(open(p + '/smoke.json', encoding='utf8')) for p in sys.argv[1:3])
ratio = b['seconds_per_update'] / a['seconds_per_update']
ok = ratio <= 1.15 and b['peak_reserved_gib'] <= 7.8
print('PASS' if ok else 'FAIL', 'A %.2f s, B_ds %.2f s (%.2fx), peak reserved A %.2f / B_ds %.2f GiB' % (
    a['seconds_per_update'], b['seconds_per_update'], ratio, a['peak_reserved_gib'], b['peak_reserved_gib']))" "$1" "$2" 2>/dev/null
}

log "E-RM Asama 1 surdurme bekliyor: GPU'da baska uzun kosu kalmayinca baslar (kapi yeniden olculmez, kayitli G0.4b okunur)"
until [ "$(busy)" = "0" ]; do sleep 60; done
step=$("$PY" -B -c "import torch; b = torch.load('$INIT', map_location='cpu', weights_only=False); print(b.get('step'))" 2>/dev/null)
if [ "$step" != "30518" ]; then log "HATA Asama 1: baslangic agirligi adim '$step', 30518 bekleniyordu"; exit 1; fi
g=$(gate "$OUT/smoke-A" "$OUT/smoke-B_ds-ut")
case "$g" in
  PASS*) D=256 ;;
  *) g=$(gate "$OUT/smoke-A" "$OUT/smoke-B_ds-ut-d128")
     case "$g" in PASS*) D=128 ;; *) log "HATA Asama 1 surdurme: kayitli kapi sonucu gecmiyor ya da okunamadi (${g:-yok})"; exit 1 ;; esac ;;
esac
log "Asama 1 surduruluyor (kayitli kapi: $g; d = $D)"

run() {  # arm seed steps
  local id="$1-s$2" extra=""
  if [ "$1" = "B_ds" ]; then extra="--ut-chunk $UT --d $D"; fi
  [ -f "$OUT/$id/final.json" ] && return 0
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "E-RM1-ut-$id" "ERM-stage1-ut" claude-4797b78c "python -B -m mline.rltm_train --arm $1 --init $INIT --seed $2 --steps $3 --out $OUT/$id $extra" "$P" "{\"seed\": $2}" "$0" >> "$LOG" 2>&1 \
    || log "MANIFEST HATA $id"
  log "BASLA E-RM Asama 1 $id ($3 guncelleme, GPU${extra:+, $extra}; kontrol noktasi varsa ondan surer)"
  "$PY" -B -m mline.rltm_train --arm "$1" --init "$INIT" --seed "$2" --steps "$3" --out "$OUT/$id" $extra >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id/final.json" ]; then log "BITTI E-RM Asama 1 $id"; else log "HATA E-RM Asama 1 $id (bkz $OUT/$id.log)"; fi
}

run B_ds 1 4000; run A 1 4000; run B_ds 2 4000
now=$(d2 4000); before=$(d2 3000)
extend=$("$PY" -B -c "import sys; n, b = sys.argv[1], sys.argv[2]; print('YES' if n != 'NA' and b != 'NA' and 0.2 <= float(n) < 0.8 and float(n) - float(b) >= 0.1 else 'NO')" "$now" "$before" 2>/dev/null)
log "Asama 1 uzatma kurali: d=2 ortanca 3000 $before -> 4000 $now: ${extend:-NO}"
if [ "$extend" = "YES" ]; then
  for id in A-s1 B_ds-s1 B_ds-s2; do [ -f "$OUT/$id/final.json" ] && mv "$OUT/$id/final.json" "$OUT/$id/final-4000.json"; done
  run B_ds 1 8000; run A 1 8000; run B_ds 2 8000
fi
log "Asama 1 okuma: $("$PY" -B -m mline.rltm_train --reading $OUT/A-s1/final.json $OUT/B_ds-s1/final.json $OUT/B_ds-s2/final.json 2>/dev/null | "$PY" -B -c "import json, sys; r = json.load(sys.stdin); print(r['label'], {k: r[k] for k in ('R1', 'R2', 'R3', 'R6', 'no_regression', 'a_at_chance')})" 2>/dev/null)"
log "E-RM Asama 1 kuyrugu BITTI"
