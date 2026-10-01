#!/usr/bin/env bash
cd /e/rlt-research/2026-09-18-phase2 || exit 1
until grep -q "ALL MODELS DONE" .superpowers/sdd/phase2/download-models.log 2>/dev/null; do sleep 20; done
"C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe" -u .superpowers/sdd/phase2/download-datasets.py > .superpowers/sdd/phase2/download-datasets.log 2>&1
