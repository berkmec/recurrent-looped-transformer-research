"""Assemble protocols/G0-pins.json from the verified model/dataset pin records, the plan snapshot hashes,
the git base and (when present) the Phase 2 venv freeze. Idempotent; re-run as artefacts arrive."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDD = ROOT / '.superpowers/sdd/phase2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    models = json.loads((SDD / 'model-pins.json').read_text(encoding='utf8')) if (SDD / 'model-pins.json').exists() else {}
    datasets = json.loads((SDD / 'dataset-pins.json').read_text(encoding='utf8')) if (SDD / 'dataset-pins.json').exists() else {}
    freeze = (SDD / 'venv-freeze.txt').read_text(encoding='utf8').splitlines() if (SDD / 'venv-freeze.txt').exists() else None
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    plan = {p.name: sha(p) for p in sorted((ROOT / 'docs/phase2-plan-snapshot').glob('*')) if p.name != 'SHA256SUMS'}
    pins = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), git_base=head,
                models={k: dict(revision=v['revision'], status=v['status'], license=v['license'], files=len(v['files']),
                                bytes=sum(f['bytes'] for f in v['files'].values()), local_dir=v['local_dir']) for k, v in models.items()},
                datasets={k: dict(revision=v['revision'], status=v['status'], license=v['license'], files=list(v['files']),
                                  bytes=sum(f['bytes'] for f in v['files'].values()), skipped=v['skipped'], local_dir=v['local_dir']) for k, v in datasets.items()},
                venv=dict(path='E:/rlt-research/.venv-phase2', freeze=freeze, note='new environment; existing venvs untouched') if freeze else 'PENDING',
                runtime_existing=dict(cuda='C:/Users/pc/Documents/ChatGPT/DaxDream/.venv (torch 2.12.0+cu130, transformers 5.8.1) used read-only for downloads and 0.6B inference'),
                plan_snapshot=plan, hardware='RTX 2060 SUPER 8 GiB, i3-10100F 4c/8t, 16 GiB RAM',
                limits='LongMemEval-M (2.6 GB) not downloaded; Turkish slices of MuSiQue/BigCodeBench do not exist; LLM answer-model calibration recorded separately in G0 report')
    out = ROOT / 'protocols/G0-pins.json'
    out.write_text(json.dumps(pins, indent=1, ensure_ascii=False) + '\n', encoding='utf8')
    print(json.dumps(dict(models={k: v['status'] for k, v in pins['models'].items()}, datasets={k: v['status'] for k, v in pins['datasets'].items()}, venv='ok' if freeze else 'PENDING', sha256=sha(out)[:16])))


if __name__ == '__main__':
    main()
