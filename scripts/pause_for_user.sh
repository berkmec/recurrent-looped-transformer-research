#!/usr/bin/env bash
# Pause every experiment at a given time today for the user's evening (2026-10-24, "yarin aksama kadar sende, yarin minecraft gunum"): waits until
# HH:MM, then scripts/pause_for_user.ps1 stops the queue runners and the experiment processes and the list goes to the queue log. Every run resumes
# from its checkpoint with scripts/resume_after_user.sh.
# Usage: nohup bash scripts/pause_for_user.sh 18:00 > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
AT="${1:-18:00}"
LOG=results/M1-finetune-distance/queue.log
log() { echo "[$(date +%H:%M:%S)] [pause] $*" >> "$LOG"; }
log "kullanici icin duraklatma $AT'de kurulu (deneyler kontrol noktasindan surer; surdurme: bash scripts/resume_after_user.sh)"
until [ "$(date +%H:%M)" = "$AT" ] || [[ "$(date +%H:%M)" > "$AT" ]]; do sleep 30; done
log "DURAKLATILIYOR ($AT): kuyruk kosuculari, sonra deney surecleri"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/pause_for_user.ps1 2>&1 | tr -d '\r' | while read -r line; do log "  $line"; done
log "DURAKLATILDI: bilgisayar kullanicinin; surdurmek icin bash scripts/resume_after_user.sh"
