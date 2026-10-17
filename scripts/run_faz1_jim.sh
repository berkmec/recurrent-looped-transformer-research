#!/usr/bin/env bash
# FAZ-1 (god emir#2 `2026-10-16T18-45-27-006Z-91b418`, muhur: notes/_hive_FAZ1_MUHUR.md): L12-E (I3-a) + A3 (I3-b) — Jim.
# SIRALI TEK SERIT: L12-E eval-only (6 mevcut Z1 ckpt) -> A3 R0 notrluk -> R1 -> R2 -> R3 -> okumalar (K4 ise R1-s2).
# On-kayitlar: notes/_hive_L12_KURAL.md (§2/§3b/§3c) + notes/_hive_A3_ONKAYIT.md (§4 matris, K1-K5). Zarflar: L12-E eval ~10 dk; A3 ≈12-15 dk (0,5 CPU-h SERT).
# I1-a: Faz-1 muhurlu paketi mevcut telemetriyle kosar — MUAF (notes/_hive_I1a_TELEMETRI.md:72); hukümde [EKSIK] alanlar isaretlenir.
# RUNMAN: her kosu basinda tek-yazim <run_id>.manifest.json (_hive_KOSU_HIJYENI_SEMA.md; god 7060c0). S3 muhur notu: 826099ce (kapali 17:48Z).
# Bu betik yalniz muhur-sonrasi commit hash'inden kosar; ajan dogrudan commit etmez (T8 emsali).
# Usage: nohup bash scripts/run_faz1_jim.sh > /dev/null 2>&1 &      (izleme: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/Z1-pilot
LOG=results/M1-finetune-distance/queue.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

manifest() {  # run_id campaign command prereg seeds   (RUNMAN tek-yazim; manifest varsa atlanir)
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "$1" "$2" jim-mv8403zi "$3" "$4" "$5" "$0" >> "$LOG" 2>&1 || log "MANIFEST HATA $1"
}

CUR=1,2,3,4,6,8,12,16
L12_CKPTS=("$OUT/perm4-hold-w64-s1-cur-egnp05.pt" "$OUT/perm4-hold-w64-s2-cur-egnp05.pt" "$OUT/perm4-hold-w64-s3-cur-egnp05.pt" \
           "$OUT/perm4-hold_nc-w64-s1-cur-egnp05.pt" "$OUT/perm5-hold-w128-s1-cur-egnp05.pt" "$OUT/perm5-hold-w128-s2-cur-egnp05.pt")

log "--- FAZ-1 kuyrugu BASLADI (Jim): S3 muhur notu 826099ce; L12-E -> A3 (R0 notrluk kapisi) -> okumalar -> (K4) R1-s2"

# ---------- I3-a / L12-E (eval-only; birincil hucre P0 x P90; tek oturum dosyasi) ----------
l12run() {  # id [extra...]
  local id="$1"; shift
  if [ -f "$OUT/$id.json" ]; then log "ATLA L12-E $id (json var)"; return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then log "ATLA L12-E $id (kilitli)"; return 0; fi
  manifest "$id" Z1-pilot "python -B -m horizon.l12_eval 6 ckpt --n 200 --seed 1 $*" "notes/_hive_L12_KURAL.md" '{"eval":1}'
  log "BASLA L12-E $id (eval-only, 1 parcacik, BelowNormal)"
  "$PY" -B -m horizon.l12_eval "${L12_CKPTS[@]}" --n 200 --seed 1 --batch 50 --out "$OUT/$id.json" "$@" >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then
    log "BITTI L12-E $id (genel: $("$PY" -B -c "import json;print(json.load(open(r'$OUT/$id.json', encoding='utf-8'))['combination']['overall'])" 2>/dev/null || echo '?'))"
  else
    log "HATA L12-E $id (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null
  fi
}
l12run l12-e-session-e1
l12run l12-e-session-e1-blocks --blocks-p90

# ---------- I3-b / A3 (matris: _hive_A3_ONKAYIT.md §4; zarf 0,5 CPU-h SERT) ----------
a3() {  # id prereg seeds cmd...
  local id="$1" prereg="$2" seeds="$3"; shift 3
  if [ -f "$OUT/$id.json" ]; then log "ATLA A3 $id (json var)"; return 0; fi
  if ! mkdir "$OUT/$id.lock" 2>/dev/null; then log "ATLA A3 $id (kilitli)"; return 0; fi
  manifest "$id" Z1-pilot "python -B -m horizon.depth_run $*" "$prereg" "$seeds"
  log "BASLA A3 $id"
  "$PY" -B -m horizon.depth_run "$@" >> "$OUT/$id.log" 2>&1
  if [ -f "$OUT/$id.json" ]; then log "BITTI A3 $id"; return 0; fi
  log "HATA A3 $id (bkz $OUT/$id.log)"; rmdir "$OUT/$id.lock" 2>/dev/null; return 1
}

a3 a3-r0-a "notes/_hive_A3_ONKAYIT.md" '{"init":1,"data_split":1}' \
  --task perm5 --arm hold --width 128 --seed 1 --steps 300 --curriculum "$CUR" --neutral-p 0 --log-every 1 \
  --gate-telemetry "$OUT/a3-r0-a.tel.jsonl" --out "$OUT/a3-r0-a.json" || exit 1
a3 a3-r0-b "notes/_hive_A3_ONKAYIT.md" '{"init":1,"data_split":1}' \
  --task perm5 --arm hold --width 128 --seed 1 --steps 300 --curriculum "$CUR" --neutral-p 0 --log-every 1 \
  --out "$OUT/a3-r0-b.json" || exit 1

# R0 kabulu: history (loss/grad_norm) bit-ozdes + guard esit. BASARISIZSA zincir DURUR (telemetri numerigi degistiriyor).
R0=$("$PY" -B -c "from horizon.gate_telemetry import compare_runs; r=compare_runs(r'$OUT/a3-r0-a.json', r'$OUT/a3-r0-b.json'); print('EQUAL' if r['equal'] else 'FAIL')" 2>/dev/null || echo HATA)
log "[A3] R0 notrluk (bit-ozdeslik): $R0"
if [ "$R0" != "EQUAL" ]; then log "[A3] ZINCIR DURDU: R0 notrluk BASARISIZ — telemetri kosuyu degistiriyor; god'a rapor"; exit 1; fi
"$PY" -B -m horizon.gate_telemetry "$OUT/a3-r0-a.tel.jsonl" >> "$OUT/a3-r0-a.schema.log" 2>&1 && log "[A3] R0-a JSONL sema-kontrolu yazildi (a3-r0-a.schema.log)"

a3 a3-r1-perm5-hold-w128-s1-np0 "notes/_hive_A3_ONKAYIT.md" '{"init":1,"data_split":1}' \
  --task perm5 --arm hold --width 128 --seed 1 --steps 6000 --curriculum "$CUR" --neutral-p 0 --log-every 1000 \
  --gate-telemetry "$OUT/a3-r1-perm5-hold-w128-s1-np0.tel.jsonl" --out "$OUT/a3-r1-perm5-hold-w128-s1-np0.json" || exit 1
a3 a3-r2-perm4-hold-w64-s1-np0 "notes/_hive_A3_ONKAYIT.md" '{"init":1,"data_split":1}' \
  --task perm4 --arm hold --width 64 --seed 1 --steps 6000 --curriculum "$CUR" --neutral-p 0 --log-every 1000 \
  --gate-telemetry "$OUT/a3-r2-perm4-hold-w64-s1-np0.tel.jsonl" --out "$OUT/a3-r2-perm4-hold-w64-s1-np0.json" || exit 1
a3 a3-r3-perm5-core-w128-s1 "notes/_hive_A3_ONKAYIT.md" '{"init":1,"data_split":1}' \
  --task perm5 --arm core --width 128 --seed 1 --steps 6000 --curriculum "$CUR" \
  --out "$OUT/a3-r3-perm5-core-w128-s1.json" || exit 1

R3LOSS=$("$PY" -B -c "import json;print(json.load(open(r'$OUT/a3-r3-perm5-core-w128-s1.json', encoding='utf-8'))['train']['history'][-1]['loss'])" 2>/dev/null || echo BILINMIYOR)
log "[A3] R3 (duz Core) son kayip: $R3LOSS (K5c esigi <= 0,5)"

# R2 okumasi (K5b: enstrumantasyon canli — kendi G_ratio'sundan)
if [ ! -f "$OUT/a3-r2-reading.json" ]; then
  "$PY" -B -m horizon.gate_telemetry --reading "$OUT/a3-r2-perm4-hold-w64-s1-np0.tel.jsonl" --ref "$OUT/a3-r2-perm4-hold-w64-s1-np0.json" \
    --r0-equal --r2-live-auto --r3-loss "$R3LOSS" --out "$OUT/a3-r2-reading.json" >> "$OUT/a3-r2-reading.log" 2>&1 \
    && log "[A3] R2 okuma yazildi: $("$PY" -B -c "import json;print(json.load(open(r'$OUT/a3-r2-reading.json', encoding='utf-8'))['label'])" 2>/dev/null || echo '?')" \
    || log "[A3] R2 okuma HATA (bkz a3-r2-reading.log)"
fi
R2LIVE=$("$PY" -B -c "import json; s=json.load(open(r'$OUT/a3-r2-reading.json', encoding='utf-8')); print('YES' if (s['G_ratio'] is not None and s['G_ratio'] >= 0.10) else 'NO')" 2>/dev/null || echo NO)
log "[A3] R2 canli mi (K5b, G_ratio >= 0,10): $R2LIVE"
R1K5=(--r0-equal --r3-loss "$R3LOSS")
if [ "$R2LIVE" = "YES" ]; then R1K5=(--r2-live "${R1K5[@]}"); else R1K5=(--r2-dead "${R1K5[@]}"); fi

# R1 okumasi (birincil; tek yazim)
if [ ! -f "$OUT/a3-r1-reading.json" ]; then
  "$PY" -B -m horizon.gate_telemetry --reading "$OUT/a3-r1-perm5-hold-w128-s1-np0.tel.jsonl" --ref "$OUT/a3-r1-perm5-hold-w128-s1-np0.json" \
    "${R1K5[@]}" --out "$OUT/a3-r1-reading.json" >> "$OUT/a3-r1-reading.log" 2>&1 \
    && log "[A3] R1 okuma yazildi: $("$PY" -B -c "import json;print(json.load(open(r'$OUT/a3-r1-reading.json', encoding='utf-8'))['label'])" 2>/dev/null || echo '?')" \
    || log "[A3] R1 okuma HATA (bkz a3-r1-reading.log)"
fi
R1L=$("$PY" -B -c "import json;print(json.load(open(r'$OUT/a3-r1-reading.json', encoding='utf-8'))['label'])" 2>/dev/null || echo BILINMIYOR)
SEED2=$("$PY" -B -c "import json;print('YES' if json.load(open(r'$OUT/a3-r1-reading.json', encoding='utf-8'))['needs_seed2'] else 'NO')" 2>/dev/null || echo NO)
log "[A3] R1 etiket: $R1L | K4 bant (needs_seed2): $SEED2"

# K4 bandindaysa R1-s2 (tohum-2) + okuma; iki etiket ayniysa o etiket, farkliysa KARISIK (tohum-ayrimi)
if [ "$SEED2" = "YES" ]; then
  a3 a3-r1-s2-perm5-hold-w128-s2-np0 "notes/_hive_A3_ONKAYIT.md" '{"init":2,"data_split":2}' \
    --task perm5 --arm hold --width 128 --seed 2 --steps 6000 --curriculum "$CUR" --neutral-p 0 --log-every 1000 \
    --gate-telemetry "$OUT/a3-r1-s2-perm5-hold-w128-s2-np0.tel.jsonl" --out "$OUT/a3-r1-s2-perm5-hold-w128-s2-np0.json" || exit 1
  if [ ! -f "$OUT/a3-r1-s2-reading.json" ]; then
    "$PY" -B -m horizon.gate_telemetry --reading "$OUT/a3-r1-s2-perm5-hold-w128-s2-np0.tel.jsonl" --ref "$OUT/a3-r1-s2-perm5-hold-w128-s2-np0.json" \
      "${R1K5[@]}" --out "$OUT/a3-r1-s2-reading.json" >> "$OUT/a3-r1-s2-reading.log" 2>&1 \
      && log "[A3] R1-s2 okuma yazildi" || log "[A3] R1-s2 okuma HATA (bkz a3-r1-s2-reading.log)"
  fi
  S2L=$("$PY" -B -c "import json;print(json.load(open(r'$OUT/a3-r1-s2-reading.json', encoding='utf-8'))['label'])" 2>/dev/null || echo BILINMIYOR)
  if [ "$R1L" = "$S2L" ]; then
    log "[A3] SONUC: R1=$R1L = R1-s2=$S2L (K4 ayni etiket; tohum-ayrimi yok)"
  else
    log "[A3] SONUC: KARISIK (tohum-ayrimi): R1=$R1L vs R1-s2=$S2L — god'a rapor"
  fi
else
  log "[A3] SONUC: R1=$R1L (K4 bant disi; R1-s2 kosulmadi)"
fi

log "--- FAZ-1 kuyrugu BITTI (Jim): L12-E + A3; I3-e (Meredith) siraya HAZIR (kilit/kuyruk bos)"
