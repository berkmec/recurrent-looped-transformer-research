"""Run manifest v1 (`notes/_hive_KOSU_HIJYENI_SEMA.md`, god-approved `7060c0`): one static provenance record per run.

Written ONCE at the start of a run ("başlangıç beyanı"), atomically (`.tmp` + `os.replace`): a half-written file never
appears in the tree. No status updates afterwards (god: no `status=completed` write; the run's .json + queue.log tell
the end). Scope: from the first run after this module lands; no backfill (transition exception for earlier runs).

Dirty convention (documented; one-line change if god wants it stricter): `dirty` counts (a) changes to TRACKED files
outside `results/` (run outputs and queue.log move during a campaign by design) and (b) untracked files under the code
roots (`horizon/`, `scripts/`, `tests/`, `core/` — new code that no seal covers yet). `porcelain` carries the COMPLETE
`git status --porcelain` list either way: nothing is hidden. `dirty=true` => the verdict can be at most KISMEN (7060c0).

One-line hook for the run scripts:
  "$PY" -B -c "import sys; from horizon.run_manifest import quick; print(quick(*sys.argv[1:]))" \
      <run_id> <campaign> <agent> "<command>" "<prereg-paper>" <seeds-json> <config-source>
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 'run-manifest/v1'
IGNORE_PREFIXES = ('results/',)                       # run outputs + queue.log move by design
CODE_PREFIXES = ('horizon/', 'scripts/', 'tests/', 'core/')
REQUIRED = ('run_id', 'campaign', 'agent', 'command', 'prereg', 'config', 'seeds')


def created_utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def classify_porcelain(lines, ignore_prefixes=IGNORE_PREFIXES, code_prefixes=CODE_PREFIXES):
    """(dirty, normalized_list) per the module docstring: tracked changes outside `results/` are dirty; untracked
    entries are dirty only under the code roots; everything stays in the returned list for the record."""
    dirty = False
    normalized = []
    for raw in lines:
        line = raw.rstrip('\n').rstrip()
        if not line:
            continue
        normalized.append(line)
        status, path = line[:2], line[3:].strip()
        path = path.rsplit(' -> ', 1)[-1].replace('\\', '/').strip('"')
        if status.strip().startswith('?'):
            if path.startswith(code_prefixes):
                dirty = True
        elif not path.startswith(ignore_prefixes):
            dirty = True
    return dirty, normalized


def git_state(root='.'):
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'],
                          capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                            capture_output=True, text=True, check=True).stdout.splitlines()
    dirty, porcelain = classify_porcelain(status)
    return dict(head=head, dirty=dirty, porcelain=porcelain)


def _sha256(path):
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_run_manifest(spec, results_root='results', out=None):
    """spec: run_id, campaign, agent, command, prereg, config{source, summary}, seeds{...}; optional: env, datasets,
    telemetry_schema (default `I1-a/v1`), code (default: collected), created_utc (default: now), modules_sha256.
    Atomic single write; an existing manifest is never overwritten. Returns the manifest Path."""
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise ValueError(f'run manifest: missing required keys {missing}')
    cfg = dict(spec['config'])
    if 'source' not in cfg or 'summary' not in cfg:
        raise ValueError('run manifest: config needs source + summary')
    cfg.setdefault('sha256', _sha256(cfg['source']))
    doc = dict(schema=SCHEMA, run_id=spec['run_id'], campaign=spec['campaign'], agent=spec['agent'],
               created_utc=spec.get('created_utc') or created_utc(), command=spec['command'],
               prereg=spec['prereg'], code=spec.get('code') or git_state(), config=cfg, seeds=spec['seeds'],
               telemetry_schema=spec.get('telemetry_schema', 'I1-a/v1'), status=spec.get('status', 'started'))
    for key in ('env', 'datasets', 'modules_sha256', 'notes'):
        if spec.get(key) is not None:
            doc[key] = spec[key]
    path = Path(out) if out else Path(results_root) / spec['campaign'] / f"{spec['run_id']}.manifest.json"
    if path.exists():
        raise FileExistsError(f'{path} exists: a run manifest is written once (baslangic beyani)')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf8')
        os.replace(tmp, path)
    finally:
        if tmp.exists():          # replace succeeded (tmp is gone) or the write failed: never leave a half file behind
            tmp.unlink()
    return path


def validate(path):
    """Structural + self-consistency check (never raises for a bad document; True/False)."""
    try:
        doc = json.loads(Path(path).read_text(encoding='utf8'))
    except Exception:
        return False
    if doc.get('schema') != SCHEMA:
        return False
    if not isinstance(doc.get('run_id'), str) or not doc['run_id']:
        return False
    for key in ('campaign', 'agent', 'created_utc', 'command'):
        if not isinstance(doc.get(key), str) or not doc.get(key):
            return False
    if not isinstance(doc.get('prereg'), dict) or not isinstance(doc.get('config'), dict) or not isinstance(doc.get('seeds'), dict):
        return False
    if 'source' not in doc['config'] or 'summary' not in doc['config'] or 'sha256' not in doc['config']:
        return False
    code = doc.get('code')
    if not isinstance(code, dict) or not isinstance(code.get('head'), str):
        return False
    if not isinstance(code.get('dirty'), bool) or not isinstance(code.get('porcelain'), list):
        return False
    expected_dirty, _ = classify_porcelain(code['porcelain'])
    if code['dirty'] != expected_dirty:
        return False
    if code['dirty'] and not code['porcelain']:
        return False
    return True


def quick(run_id, campaign, agent, command, prereg, seeds='{}', config_source=None, results_root='results'):
    """One-line hook for the run scripts: fills created_utc, code and the config sha256 itself.
    `seeds` is a JSON string (or dict). Returns the manifest path as a string, or a note when it already exists."""
    if isinstance(seeds, str):
        seeds = json.loads(seeds)
    spec = dict(run_id=run_id, campaign=campaign, agent=agent, command=command, prereg={'paper': prereg},
                config=dict(source=config_source or 'scripts', summary={}), seeds=seeds)
    try:
        return str(write_run_manifest(spec, results_root=results_root))
    except FileExistsError as e:
        return f'manifest exists: {e}'


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('manifest')
    p.add_argument('--print', dest='show', action='store_true')
    a = p.parse_args()
    ok = validate(a.manifest)
    print(f'{a.manifest}: validate={ok}')
    if a.show:
        print(Path(a.manifest).read_text(encoding='utf8'))
    raise SystemExit(0 if ok else 1)
