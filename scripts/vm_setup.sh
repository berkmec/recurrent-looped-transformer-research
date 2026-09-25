#!/usr/bin/env bash
# Provision a fresh Ubuntu + NVIDIA box for the M-line. Provider-agnostic: anything with a CUDA driver and a
# GPU works, so this makes no assumption about AWS, GCP or anyone else.
#
# WHY THIS EXISTS. The local run was stopped at 13 % because the machine ran low on memory while it was idle, and
# the campaign has lost a run to a power cut before. A rented box runs headless and uninterrupted. The budget is
# small (about 24 USD), so the whole point is that hour one is spent measuring rather than installing.
#
# Usage, on the VM:
#   bash vm_setup.sh /data/rlt            # target root; the repo and corpus go under it
#
# It is idempotent: run it again after a spot preemption and it will skip what is already in place.
set -euo pipefail

ROOT="${1:-/data/rlt}"
REPO="$ROOT/2026-09-18-phase2"
DATA="$ROOT/mline-data/cnn-bytes"

TRAIN_SHA=edeef82f388f81526652bfb7247e0dcc3f7c85ce691513d77a527aad0c17b8e4
VAL_SHA=451b7be62e1548db80160709c5cd527f5f660600f182f3a2cbf2fffa33c7ac39

say() { printf '\n== %s ==\n' "$1"; }

say "hardware"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nproc; free -g | head -2; df -h "$ROOT" 2>/dev/null || df -h /

say "system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip git rsync >/dev/null

say "python environment"
# A FRESH venv, unlike the local one. The campaign rule forbids installing into the sealed local venv; it says
# nothing about a machine that did not exist yesterday. Torch is pinned to the local major/minor so the arms stay
# comparable with anything already measured here.
if [ ! -d "$ROOT/venv" ]; then
  python3.11 -m venv "$ROOT/venv"
fi
"$ROOT/venv/bin/pip" install -q --upgrade pip
"$ROOT/venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu124
"$ROOT/venv/bin/pip" install -q numpy pytest

say "versions"
"$ROOT/venv/bin/python" - <<'PY'
import sys, torch
print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda)
print('gpu    ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - stop here')
print('triton ', end='')
try:
    import triton; print(triton.__version__, '(torch.compile available - the one thing Windows could not give us)')
except ImportError:
    print('missing; torch.compile will not help')
PY

say "corpus integrity"
# The corpus is the experiment's ground truth. A silent truncation during upload would change every result while
# looking like a successful transfer, so the bytes are verified, not trusted.
for pair in "train.bin:$TRAIN_SHA" "validation.bin:$VAL_SHA"; do
  file="${pair%%:*}"; want="${pair##*:}"
  if [ ! -f "$DATA/$file" ]; then echo "MISSING $DATA/$file - upload it first"; exit 1; fi
  got=$(sha256sum "$DATA/$file" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then echo "CHECKSUM MISMATCH $file"; echo "  want $want"; echo "  got  $got"; exit 1; fi
  echo "ok $file $(stat -c %s "$DATA/$file") bytes"
done

say "test suite"
cd "$REPO"
MLINE_DATA="$DATA" "$ROOT/venv/bin/python" -B tests/run_all.py

say "step cost, measured before anything long is launched"
# Never assume the rented card is faster than the one at home. This campaign's speed estimates have been wrong by
# 3x, 5x and 13x, always optimistic, so the schedule is rebuilt from THIS number rather than from a prediction.
MLINE_DATA="$DATA" "$ROOT/venv/bin/python" -B -m mline.train --probe

echo
echo "setup complete. Next: bash scripts/vm_run.sh $ROOT"
