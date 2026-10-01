#!/usr/bin/env bash
LOG=/e/rlt-research/2026-09-18-phase2/.superpowers/sdd/phase2/venv-install.log
V=/e/rlt-research/.venv-phase2
{
echo "$(date -u +%FT%TZ) START venv"
"C:/Program Files/Python311/python.exe" -m venv "$V" || { echo "VENV FAILED"; exit 2; }
"$V/Scripts/python.exe" -m pip install --upgrade pip 2>&1 | tail -1
echo "$(date -u +%FT%TZ) torch cu130"
"$V/Scripts/python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -3
echo "$(date -u +%FT%TZ) libs"
"$V/Scripts/python.exe" -m pip install transformers accelerate bitsandbytes datasets huggingface_hub rank_bm25 safetensors numpy scipy 2>&1 | tail -3
echo "$(date -u +%FT%TZ) verify"
"$V/Scripts/python.exe" -c "import torch,transformers,accelerate,bitsandbytes,datasets,rank_bm25;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'| transformers',transformers.__version__,'| bnb',bitsandbytes.__version__,'| datasets',datasets.__version__)"
"$V/Scripts/python.exe" -m pip freeze > /e/rlt-research/2026-09-18-phase2/.superpowers/sdd/phase2/venv-freeze.txt
echo "$(date -u +%FT%TZ) DONE venv"
} >> "$LOG" 2>&1
