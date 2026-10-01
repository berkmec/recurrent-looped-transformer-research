#!/usr/bin/env bash
LOG=/e/rlt-research/2026-09-18-phase2/.superpowers/sdd/phase2/venv-fix-torch.log
V=/e/rlt-research/.venv-phase2
export TMP=E:/rlt-research/tmp TEMP=E:/rlt-research/tmp TMPDIR=E:/rlt-research/tmp
{
echo "$(date -u +%FT%TZ) reinstall torch cu130 (temp on E:)"
"$V/Scripts/python.exe" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.0+cu130" 2>&1 | tail -4
echo "$(date -u +%FT%TZ) verify"
"$V/Scripts/python.exe" -c "import torch,transformers,bitsandbytes,accelerate;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'| transformers',transformers.__version__,'| bnb',bitsandbytes.__version__)"
"$V/Scripts/python.exe" -m pip freeze > /e/rlt-research/2026-09-18-phase2/.superpowers/sdd/phase2/venv-freeze.txt
echo "$(date -u +%FT%TZ) DONE torch fix"
} >> "$LOG" 2>&1
