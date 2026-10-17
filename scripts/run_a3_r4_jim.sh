#!/usr/bin/env bash
# A3 R4 (god fab781, 2026-10-17): muhurlu E-G hucresi replikasyonu — perm5 hold w128 np0.5 30k, 1 tohum.
# Dayanak: notes/_hive_A3_ONKAYIT.md §4 (R4 satiri) + §6 karar maddesi; zarf <=0,4 CPU-h SERT (olculen ~0,36; 15 dk ustu marj onayi: god 78b74f).
# Sira: I3-d (Meredith) + P0 rerun (Dwight) SONRASI — tek serit; serit teyidi bende. Bu betik yalniz muhur-sonrasi commit hash'inden kosar.
# RUNMAN: tek-yazim <run_id>.manifest.json (dirty=false beklenir); ciktilar results/Z1-pilot/.
# Usage: nohup bash scripts/run_a3_r4_jim.sh > /dev/null 2>&1 &   (izleme: results/M1-finetune-distance/queue.log)
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
ID=a3-r4-perm5-hold-w128-s1-np05-30k

log "--- A3 R4 BASLIYOR (Jim): muhurlu E-G hucre replikasyonu (perm5 hold w128 np0.5 30k); god fab781; tek serit"

# On-kosul: zincir kanitlari (K5 girdileri) yerinde ve okunur mu — degilse kosmadan dur.
R3LOSS=$("$PY" -B -c "import json;print(json.load(open(r'$OUT/a3-r3-perm5-core-w128-s1.json', encoding='utf-8'))['train']['history'][-1]['loss'])" 2>/dev/null || echo BILINMIYOR)
R2LIVE=$("$PY" -B -c "import json; s=json.load(open(r'$OUT/a3-r2-reading.json', encoding='utf-8')); print('YES' if (s['G_ratio'] is not None and s['G_ratio'] >= 0.10) else 'NO')" 2>/dev/null || echo BILINMIYOR)
if [ "$R3LOSS" = "BILINMIYOR" ] || [ "$R2LIVE" = "BILINMIYOR" ]; then log "HATA A3 R4 on-kosul: zincir kanitlari okunamadi (R3LOSS=$R3LOSS R2LIVE=$R2LIVE)"; exit 1; fi
log "[A3-R4] on-kosul OK: R3LOSS=$R3LOSS R2LIVE=$R2LIVE"

run4() {
  if [ -f "$OUT/$ID.json" ]; then log "ATLA A3 R4 $ID (json var)"; return 0; fi
  if ! mkdir "$OUT/$ID.lock" 2>/dev/null; then log "ATLA A3 R4 $ID (kilitli)"; return 0; fi
  manifest "$ID" Z1-pilot "python -B -m horizon.depth_run $* --task perm5 --arm hold --width 128 --seed 1 --steps 30000 --curriculum $CUR --neutral-p 0.5 --log-every 1000 --gate-telemetry $OUT/$ID.tel.jsonl --out $OUT/$ID.json" "notes/_hive_A3_ONKAYIT.md" '{"init":1,"data_split":1}'
  log "BASLA A3 R4 $ID (30k adim, BelowNormal)"
  "$PY" -B -m horizon.depth_run --task perm5 --arm hold --width 128 --seed 1 --steps 30000 --curriculum "$CUR" --neutral-p 0.5 --log-every 1000 \
    --gate-telemetry "$OUT/$ID.tel.jsonl" --out "$OUT/$ID.json" >> "$OUT/$ID.log" 2>&1
  if [ -f "$OUT/$ID.json" ]; then log "BITTI A3 R4 $ID"; return 0; fi
  log "HATA A3 R4 $ID (bkz $OUT/$ID.log)"; rmdir "$OUT/$ID.lock" 2>/dev/null; return 1
}

run4 || exit 1

# Okuma (K5 zincir kanitlariyla; tek yazim)
K5FLAGS=(--r0-equal --r3-loss "$R3LOSS")
if [ "$R2LIVE" = "YES" ]; then K5FLAGS=(--r2-live "${K5FLAGS[@]}"); else K5FLAGS=(--r2-dead "${K5FLAGS[@]}"); fi
if [ ! -f "$OUT/$ID-reading.json" ]; then
  "$PY" -B -m horizon.gate_telemetry --reading "$OUT/$ID.tel.jsonl" --ref "$OUT/$ID.json" \
    "${K5FLAGS[@]}" --out "$OUT/$ID-reading.json" >> "$OUT/$ID-reading.log" 2>&1 \
    && log "[A3-R4] okuma yazildi: $("$PY" -B -c "import json;print(json.load(open(r'$OUT/$ID-reading.json', encoding='utf-8'))['label'])" 2>/dev/null || echo '?')" \
    || log "[A3-R4] okuma HATA (bkz $ID-reading.log)"
fi
LABEL=$("$PY" -B -c "import json;print(json.load(open(r'$OUT/$ID-reading.json', encoding='utf-8'))['label'])" 2>/dev/null || echo BILINMIYOR)
log "--- A3 R4 BITTI (Jim): etiket: $LABEL; cikti=$ID.json; manifest+okuma hazir — I3-c karari god'a"
