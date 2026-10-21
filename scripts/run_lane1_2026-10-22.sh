#!/usr/bin/env bash
# CPU lane 1 queue, 2026-10-22 (user 2026-10-21: "GPU CPU senindir, koşuları başlat"). Waits for the A3 R4 run on this lane
# to finish, then one process at a time, BelowNormal, one thread, a RUNMAN manifest per run:
#   1. P6, descriptive (E-R5 registered use, note 5.31.2): drift profile of S5 Core seed 3; margin profile of seeds 1, 2, 3.
#   2. I3-f (notes/_hive_F2_ONKAYIT_I3-f.md, sealed, committed ae15d94 before this runs): cells F1..F5, each after a
#      10-step dry run of its exact command (the paper asks for one); a failed dry run skips I3-f and is logged.
#   3. E-F5 (note 5.24): the three runs cut on 2026-10-02 (alpha 1.0, peak lr 1e-3) and by the 2026-10-15 reboot (alpha 0.3).
# Usage: nohup bash scripts/run_lane1_2026-10-22.sh > /dev/null 2>&1 &     (progress: results/M1-finetune-distance/queue.log)
set -u
cd "$(dirname "$0")/.."

WINPID=$(cat /proc/$$/winpid 2>/dev/null || true)
if [ -n "${WINPID:-}" ]; then powershell.exe -NoProfile -Command "(Get-Process -Id $WINPID).PriorityClass = 'BelowNormal'" > /dev/null 2>&1 || true; fi

export PY="__PY_ENV_DIR__/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=results/M1-finetune-distance/queue.log
Z=results/Z1-pilot
CUR=1,2,3,4,6,8,12,16
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
manifest() {  # run_id campaign command prereg seeds
  "$PY" -B -c "from horizon.run_manifest import quick; import sys; print(quick(*sys.argv[1:]))" \
    "$1" "$2" claude-4797b78c "$3" "$4" "$5" "$0" >> "$LOG" 2>&1 || log "MANIFEST HATA $1"
}

log "LANE1 kuyrugu bekliyor: A3 R4 bitisi"
until grep -q -E "A3 R4 BITTI|HATA A3 R4" "$LOG"; do sleep 30; done
log "LANE1 kuyrugu basliyor (P6 -> I3-f -> E-F5)"

# 1. P6 (descriptive)
if [ ! -f "$Z/drift-profile-s3.json" ]; then
  manifest P6-drift-profile-s3 Z1-pilot "python -B -m horizon.drift_profile $Z/perm5-core-w128-s3-cur.pt --out $Z/drift-profile-s3.json" "notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.31.2" '{"seed": 3}'
  "$PY" -B -m horizon.drift_profile "$Z/perm5-core-w128-s3-cur.pt" --out "$Z/drift-profile-s3.json" > "$Z/drift-profile-s3.txt" 2>&1 \
    && log "BITTI P6 drift-profile-s3" || log "HATA P6 drift-profile-s3 (bkz $Z/drift-profile-s3.txt)"
fi
if [ ! -f "$Z/margin-profile-s5-seeds.json" ]; then
  manifest P6-margin-profile-s5-seeds Z1-pilot "python -B -m horizon.margin_profile $Z/perm5-core-w128-s1-cur.pt $Z/perm5-core-w128-s2-cur.pt $Z/perm5-core-w128-s3-cur.pt --out $Z/margin-profile-s5-seeds.json" "notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.31.2" '{"seeds": [1, 2, 3]}'
  "$PY" -B -m horizon.margin_profile "$Z/perm5-core-w128-s1-cur.pt" "$Z/perm5-core-w128-s2-cur.pt" "$Z/perm5-core-w128-s3-cur.pt" \
    --out "$Z/margin-profile-s5-seeds.json" > "$Z/margin-profile-s5-seeds.txt" 2>&1 \
    && log "BITTI P6 margin-profile-s5-seeds" || log "HATA P6 margin-profile-s5-seeds (bkz $Z/margin-profile-s5-seeds.txt)"
fi

# 2. I3-f
i3f_args() { echo "--task $1 --arm $2 --width $3 --seed 1 --curriculum $CUR --neutral-p 0.9 --hold-tmax 16"; }
CELLS="F1:perm4:core:64 F2:perm4:hold:64 F3:perm4:hold_nc:64 F4:perm5:core:128 F5:perm5:hold:128"
DRY=results/F2-I3f-dryrun; mkdir -p "$DRY"; dry_ok=1
for c in $CELLS; do
  IFS=: read -r cell task arm width <<< "$c"
  $PY -B -m horizon.depth_run $(i3f_args "$task" "$arm" "$width") --steps 10 --out "$DRY/$cell.json" > "$DRY/$cell.log" 2>&1 || dry_ok=0
  [ -f "$DRY/$cell.json" ] || dry_ok=0
done
if [ "$dry_ok" = 1 ]; then
  log "I3-f kuru deneme (10 adim) 5/5 hucre tamam"
  for c in $CELLS; do
    IFS=: read -r cell task arm width <<< "$c"
    id="F2-I3f-$cell-$task-$arm-w$width-s1-np09"
    [ -f "results/$id.json" ] && continue
    mkdir "results/$id.lock" 2>/dev/null || continue
    cmd="python -B -m horizon.depth_run $(i3f_args "$task" "$arm" "$width") --steps 15000 --log-every 1000 --out results/$id.json"
    manifest "$id" F2-I3f "$cmd" "notes/_hive_F2_ONKAYIT_I3-f.md" '{"seed": 1}'
    log "BASLA I3-f $id (CPU, 1 parcacik, 15000 adim, notr-p 0.9, hold-tmax 16)"
    $PY -B -m horizon.depth_run $(i3f_args "$task" "$arm" "$width") --steps 15000 --log-every 1000 --out "results/$id.json" >> "results/$id.log" 2>&1
    if [ -f "results/$id.json" ]; then log "BITTI I3-f $id"; else log "HATA I3-f $id (bkz results/$id.log)"; rmdir "results/$id.lock" 2>/dev/null; fi
  done
else
  log "HATA I3-f kuru deneme basarisiz (bkz $DRY/*.log): I3-f atlandi"
fi

# 3. E-F5 (the three cut runs; the registered runner's exact command, plus a manifest)
ef5() {  # suffix lr alpha
  local id="perm5-core-w128-s1-cur-lrdecay0.3-$1"
  [ -f "$Z/$id.json" ] && return 0
  [ -d "$Z/$id.lock" ] && rmdir "$Z/$id.lock" 2>/dev/null
  mkdir "$Z/$id.lock" 2>/dev/null || return 0
  local cmd="python -B -m horizon.depth_run --task perm5 --arm core --width 128 --seed 1 --steps 30000 --lr $2 --alpha $3 --curriculum $CUR --lr-decay-fraction 0.3 --out $Z/$id.json"
  manifest "$id" Z1-pilot "$cmd" "notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md 5.24" '{"seed": 1}'
  echo "--- start $(date '+%Y-%m-%d %H:%M:%S')" >> "$Z/$id.log"
  log "BASLA Z1 $id (E-F5; CPU, 1 parcacik, 30000 adim, tepe lr $2, alfa $3, lr son %30'da ~0'a, mufredat $CUR)"
  $PY -B -m horizon.depth_run --task perm5 --arm core --width 128 --seed 1 --steps 30000 --lr "$2" --alpha "$3" --curriculum "$CUR" \
    --lr-decay-fraction 0.3 --out "$Z/$id.json" --log-every 2000 >> "$Z/$id.log" 2>&1
  if [ -f "$Z/$id.json" ]; then log "BITTI Z1 $id"; else log "HATA Z1 $id (bkz $Z/$id.log)"; rmdir "$Z/$id.lock" 2>/dev/null; fi
}
ef5 alpha1.0 0.003 1.0
ef5 lr0.001 0.001 0.1
ef5 alpha0.3 0.003 0.3
log "LANE1 kuyrugu BITTI"
