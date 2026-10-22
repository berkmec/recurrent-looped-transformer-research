#!/usr/bin/env bash
# CPU queue 2026-10-22 b (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.42 b-d): E-S stage B seed 2 (five arms, rule 5.16
# unchanged), A3 R4 seed 2 (notes/_hive_A3_ONKAYIT.md K4), E-S-LR (sk4 GRU / LSTM / transformer at peak lr 1e-3, seeds 1-2, rule 5.42 c).
# Work-stealing: start TWO copies; each walks the same list and takes the first job that has neither a result nor a lock (atomic
# mkdir), so two CPU processes run at a time. One thread per process, BelowNormal, a RUNMAN manifest per run.
#
# Usage: nohup bash scripts/run_queue_2026-10-22b.sh > /dev/null 2>&1 &   (twice; progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=results/M1-finetune-distance/queue.log
P=notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md
mkdir -p results/ES-pilot results/ES-lr
log() { echo "[$(date +%H:%M:%S)] [q22b] $*" >> "$LOG"; }
manifest() {  # run_id campaign command prereg seeds
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "$1" "$2" claude-4797b78c "$3" "$4" "$5" "$0" >> "$LOG" 2>&1 || log "MANIFEST HATA $1"
}

sk() {   # dir task arm width seed lr(default|value) prereg
  local dir="$1" task="$2" arm="$3" width="$4" seed="$5" lr="$6" prereg="$7"
  local id="$task-$arm-w$width-s$seed" lrflag=() shown=0.003
  if [ "$lr" != default ]; then id="$id-lr$lr"; lrflag=(--lr "$lr"); shown="$lr"; fi
  [ -f "$dir/$id.json" ] && return 0
  mkdir "$dir/$id.lock" 2>/dev/null || return 0
  manifest "$id" "$(basename "$dir")" "python -B -m horizon.sk_run --task $task --arm $arm --width $width --seed $seed ${lrflag[*]:-} --steps 15000 --out $dir/$id.json --log-every 1000" "$prereg" "{\"seed\": $seed}"
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$dir/$id.log"
  log "BASLA ES $id (CPU, 1 parcacik, 15000 adim, tepe lr $shown, son %30'da ~0'a, mufredat 0..16)"
  "$PY" -B -m horizon.sk_run --task "$task" --arm "$arm" --width "$width" --seed "$seed" ${lrflag[@]+"${lrflag[@]}"} --steps 15000 \
    --out "$dir/$id.json" --log-every 1000 >> "$dir/$id.log" 2>&1
  if [ -f "$dir/$id.json" ]; then log "BITTI ES $id"; else log "HATA ES $id (bkz $dir/$id.log)"; rmdir "$dir/$id.lock" 2>/dev/null; fi
}

r4s2() {  # A3 R4 seed 2: the R4 command (scripts/run_a3_r4_jim.sh) with --seed 2, then the same reading
  local OUT=results/Z1-pilot ID=a3-r4-perm5-hold-w128-s2-np05-30k CUR=1,2,3,4,6,8,12,16
  [ -f "$OUT/$ID-reading.json" ] && return 0
  if [ ! -f "$OUT/$ID.json" ]; then
    mkdir "$OUT/$ID.lock" 2>/dev/null || return 0
    manifest "$ID" Z1-pilot "python -B -m horizon.depth_run --task perm5 --arm hold --width 128 --seed 2 --steps 30000 --curriculum $CUR --neutral-p 0.5 --log-every 1000 --gate-telemetry $OUT/$ID.tel.jsonl --out $OUT/$ID.json" "notes/_hive_A3_ONKAYIT.md K4; $P 5.42 d" '{"seed": 2}'
    log "BASLA A3 R4-s2 $ID (30k adim, BelowNormal; K4 bandi: R4 H_final 0.413)"
    "$PY" -B -m horizon.depth_run --task perm5 --arm hold --width 128 --seed 2 --steps 30000 --curriculum "$CUR" --neutral-p 0.5 --log-every 1000 \
      --gate-telemetry "$OUT/$ID.tel.jsonl" --out "$OUT/$ID.json" >> "$OUT/$ID.log" 2>&1
    if [ ! -f "$OUT/$ID.json" ]; then log "HATA A3 R4-s2 $ID (bkz $OUT/$ID.log)"; rmdir "$OUT/$ID.lock" 2>/dev/null; return 1; fi
    log "BITTI A3 R4-s2 $ID"
  fi
  mkdir "$OUT/$ID-reading.lock" 2>/dev/null || return 0
  local R3LOSS R2LIVE K5FLAGS
  R3LOSS=$("$PY" -B -c "import json;print(json.load(open(r'$OUT/a3-r3-perm5-core-w128-s1.json', encoding='utf-8'))['train']['history'][-1]['loss'])" 2>/dev/null || echo BILINMIYOR)
  R2LIVE=$("$PY" -B -c "import json; s=json.load(open(r'$OUT/a3-r2-reading.json', encoding='utf-8')); print('YES' if (s['G_ratio'] is not None and s['G_ratio'] >= 0.10) else 'NO')" 2>/dev/null || echo BILINMIYOR)
  if [ "$R3LOSS" = BILINMIYOR ] || [ "$R2LIVE" = BILINMIYOR ]; then log "HATA A3 R4-s2 okuma on-kosul (R3LOSS=$R3LOSS R2LIVE=$R2LIVE)"; return 1; fi
  K5FLAGS=(--r0-equal --r3-loss "$R3LOSS")
  if [ "$R2LIVE" = YES ]; then K5FLAGS=(--r2-live "${K5FLAGS[@]}"); else K5FLAGS=(--r2-dead "${K5FLAGS[@]}"); fi
  "$PY" -B -m horizon.gate_telemetry --reading "$OUT/$ID.tel.jsonl" --ref "$OUT/$ID.json" "${K5FLAGS[@]}" --out "$OUT/$ID-reading.json" >> "$OUT/$ID-reading.log" 2>&1 \
    && log "[A3-R4-s2] okuma yazildi: $("$PY" -B -c "import json;print(json.load(open(r'$OUT/$ID-reading.json', encoding='utf-8'))['label'])" 2>/dev/null || echo '?')" \
    || log "[A3-R4-s2] okuma HATA (bkz $OUT/$ID-reading.log)"
}

B="$P 5.16 + 5.42 b"; C="$P 5.42 c"
sk results/ES-pilot sk4 core 64 2 default "$B"
r4s2
sk results/ES-pilot sk4 lstm 112 2 default "$B"
sk results/ES-pilot sk4 gru 128 2 default "$B"
sk results/ES-pilot sk4 core_nofb 64 2 default "$B"
sk results/ES-pilot sk4 tfm 64 2 default "$B"
sk results/ES-lr sk4 gru 128 1 0.001 "$C"
sk results/ES-lr sk4 gru 128 2 0.001 "$C"
sk results/ES-lr sk4 lstm 112 1 0.001 "$C"
sk results/ES-lr sk4 lstm 112 2 0.001 "$C"
sk results/ES-lr sk4 tfm 64 1 0.001 "$C"
sk results/ES-lr sk4 tfm 64 2 0.001 "$C"
log "kopya $$ listeyi bitirdi"
