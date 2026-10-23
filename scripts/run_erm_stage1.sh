#!/usr/bin/env bash
# E-RM stage 1 v2 (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.61): L1's final weights (30,518 updates) with the
# shifted-key delta memory (B_ds, seeds 1-2) and without it (A, seed 1), 4,000 updates of note 5.40's mixture, on the GPU right after
# E-L1C (user priority 2026-10-23 13:0x, "RLT first": this queue moved ahead of E-X1, and B_ds seed 1, the L1 + memory model, runs first).
# Gate G0.4 first: 20-update smoke tests of A and B_ds; B_ds must stay within 1.15x of A's seconds per update and under 7.8 GiB of
# reserved memory, or nothing runs. One extension rule (note 5.40): at 4,000, if the B_ds seeds' median exact accuracy at d = 2
# (held-out real words) is in [0.2, 0.8) and rose by >= 0.1 since 3,000, every run continues to 8,000.
# Usage: nohup bash scripts/run_erm_stage1.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
OUT=results/ERM-stage1
LOG=results/M1-finetune-distance/queue.log
P="notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.61"
INIT=results/L1-scratch-w928-s401/latest.pt
log() { echo "[$(date +%H:%M:%S)] [erm1] $*" >> "$LOG"; }
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

log "E-RM Asama 1 bekliyor: E-L1C egitimi bitince (okuma satiri) ve GPU'da baska uzun kosu kalmayinca baslar (E-X1'in onune alindi)"
until grep -q -E "\[l1c\] (BITTI|HATA) E-L1C okuma" "$LOG"; do sleep 120; done
until [ "$(busy)" = "0" ]; do sleep 60; done
step=$("$PY" -B -c "import torch; b = torch.load('$INIT', map_location='cpu', weights_only=False); print(b.get('step'))" 2>/dev/null)
if [ "$step" != "30518" ]; then log "HATA Asama 1: baslangic agirligi adim '$step', 30518 bekleniyordu"; exit 1; fi
mkdir -p "$OUT"

for arm in A B_ds; do
  "$PY" -B -m mline.rltm_train --arm "$arm" --init "$INIT" --seed 1 --out "$OUT/smoke-$arm" --smoke 20 >> "$OUT/smoke-$arm.log" 2>&1 \
    || log "HATA G0.4 duman testi $arm (bkz $OUT/smoke-$arm.log)"
done
gate=$("$PY" -B -c "
import json
a, b = (json.load(open('$OUT/smoke-%s/smoke.json' % k, encoding='utf8')) for k in ('A', 'B_ds'))
ratio = b['seconds_per_update'] / a['seconds_per_update']
ok = ratio <= 1.15 and b['peak_reserved_gib'] <= 7.8
print('PASS' if ok else 'FAIL', 'A %.2f s, B_ds %.2f s (%.2fx), peak reserved A %.2f / B_ds %.2f GiB' % (
    a['seconds_per_update'], b['seconds_per_update'], ratio, a['peak_reserved_gib'], b['peak_reserved_gib']))" 2>/dev/null)
log "G0.4: ${gate:-olculemedi}"
case "$gate" in PASS*) ;; *) log "Asama 1 DURDU: G0.4 gecmedi (on-kayitli yedek: d = 128 ya da write_chunk 16, ayri karar)"; exit 1;; esac

run() {  # arm seed steps
  local id="$1-s$2"
  [ -f "$OUT/$id/final.json" ] && return 0
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "E-RM1-$id" "ERM-stage1" claude-4797b78c "python -B -m mline.rltm_train --arm $1 --init $INIT --seed $2 --steps $3 --out $OUT/$id" "$P" "{\"seed\": $2}" "$0" >> "$LOG" 2>&1 \
    || log "MANIFEST HATA $id"
  log "BASLA E-RM Asama 1 $id ($3 guncelleme, GPU)"
  "$PY" -B -m mline.rltm_train --arm "$1" --init "$INIT" --seed "$2" --steps "$3" --out "$OUT/$id" >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id/final.json" ]; then log "BITTI E-RM Asama 1 $id"; else log "HATA E-RM Asama 1 $id (bkz $OUT/$id.log)"; fi
}

run B_ds 1 4000; run A 1 4000; run B_ds 2 4000
now=$(d2 4000); before=$(d2 3000)
extend=$("$PY" -B -c "import sys; n, b = sys.argv[1], sys.argv[2]; print('YES' if n != 'NA' and b != 'NA' and 0.2 <= float(n) < 0.8 and float(n) - float(b) >= 0.1 else 'NO')" "$now" "$before" 2>/dev/null)
log "Asama 1 uzatma kurali: d=2 ortanca 3000 $before -> 4000 $now: ${extend:-NO}"
if [ "$extend" = "YES" ]; then
  for id in A-s1 B_ds-s1 B_ds-s2; do mv "$OUT/$id/final.json" "$OUT/$id/final-4000.json"; done
  run B_ds 1 8000; run A 1 8000; run B_ds 2 8000
fi
log "Asama 1 okuma: $("$PY" -B -m mline.rltm_train --reading $OUT/A-s1/final.json $OUT/B_ds-s1/final.json $OUT/B_ds-s2/final.json 2>/dev/null | "$PY" -B -c "import json, sys; r = json.load(sys.stdin); print(r['label'], {k: r[k] for k in ('R1', 'R2', 'R3', 'R6', 'no_regression', 'a_at_chance')})" 2>/dev/null)"
log "E-RM Asama 1 kuyrugu BITTI"
