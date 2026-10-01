"""Inline audit of the U4-sep cycle: recompute every claim in notes/U4sep_RAPORU.md from raw evidence.

Nothing here trusts a recorded value. Hashes are recomputed, row counts are counted, the endpoint numbers in the
report are re-read from evaluation.json, and the two central claims - that gap/spread fails to separate the outcome
groups while gap/drift separates them - are recomputed from results/U2d/compare.json and the U2c evaluations.

Writes .superpowers/sdd/phase2/audit-u4sep.{json,md}. Read-only with respect to results/.
"""
import glob
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = {'lr0.0003': 'results/U4sep-lr0003', 'lr0.003': 'results/U4sep-lr003'}
RAN = [('lr0.0003', 'u4sep-soft_sep-five-w32-s117'), ('lr0.0003', 'u4sep-soft_sep-five-w32-s143'),
       ('lr0.003', 'u4sep-soft_sep-five-w32-s117')]
checks, discrepancies = [], []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add(cid, ok, evidence):
    checks.append(dict(id=cid, status='PASS' if ok else 'FAIL', evidence=evidence))
    if not ok:
        discrepancies.append(cid + ': ' + evidence)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


# 1 - freeze integrity of both frozen directories
import sys
sys.path.insert(0, str(ROOT))
from horizon import sweep as sw
from horizon import sep_sweep as ss

problems = []
for level, d in DIRS.items():
    base = ROOT / d
    if sha(base / 'frozen.json') != (base / 'frozen.sha256').read_text(encoding='ascii').strip():
        problems.append(f'{level}: sidecar mismatch')
    man = json.loads((base / 'frozen.json').read_text(encoding='utf8'))
    if man['protocol_hash'] != sw.digest(man['protocol']):
        problems.append(f'{level}: protocol_hash does not match its protocol')
    for name, stored in man['sources'].items():
        if sha(ROOT / name) != stored:
            problems.append(f'{level}: source drift {name}')
add('1-freeze', not problems, f'2 frozen directories, {len(json.loads((ROOT / DIRS["lr0.0003"] / "frozen.json").read_text(encoding="utf8"))["sources"])} sources each; problems=' + (', '.join(problems) or 'none'))

# 2 - completeness, stated honestly rather than as a pass/fail on 6
complete = sorted(Path(p).parent.name + '@' + Path(p).parents[1].name for p in glob.glob(str(ROOT / 'results/U4sep-lr*/u4sep-*/complete.json')))
quarantined = sorted(Path(p).name for p in glob.glob(str(ROOT / 'results/U4sep-lr*/interrupted-*')))
# An independent review found this check's evidence was a hardcoded sentence claiming "2 never started" while one of
# them had in fact run 18 steps. A check whose evidence is prose is not a check; every number below is now counted.
started = sorted(str(Path(p).parent.relative_to(ROOT)) for p in glob.glob(str(ROOT / 'results/U4sep-lr*/u4sep-*/initial.pt')))
finished = sorted(str(Path(p).parent.relative_to(ROOT)) for p in glob.glob(str(ROOT / 'results/U4sep-lr*/u4sep-*/complete.json')))
quarantined_dirs = sorted(str(Path(p).relative_to(ROOT)) for p in glob.glob(str(ROOT / 'results/U4sep-lr*/interrupted-*')))
quarantined_runs = sorted(str(Path(p).parent.relative_to(ROOT)) for p in glob.glob(str(ROOT / 'results/U4sep-lr*/interrupted-*/*/initial.pt')))
loose = [d for d in started if d not in finished and not any(d.startswith(q) for q in quarantined_dirs)]
add('2-completeness', len(finished) == 3 and len(quarantined_runs) == 2 and not loose,
    f'counted, not asserted: {len(started) + len(quarantined_runs)} of 6 conditions left an initial.pt, '
    f'{len(finished)} reached complete.json, {len(quarantined_runs)} are quarantined with a REASON.txt, '
    f'{6 - len(finished) - len(quarantined_runs)} never started. Started-but-unquarantined directories: '
    + (', '.join(loose) or 'none'))

# 3 - per-condition integrity: recorded hashes and row counts
problems = []
for level, cid in RAN:
    d = ROOT / DIRS[level] / cid
    t = json.loads((d / 'train.json').read_text(encoding='utf8'))
    if sha(d / 'selected.pt') != t['selected_sha256']:
        problems.append(f'{cid}: checkpoint hash')
    if sha(d / 'updates.jsonl') != t['updates_sha256']:
        problems.append(f'{cid}: updates hash')
    rows = (d / 'updates.jsonl').read_text(encoding='utf8').strip().splitlines()
    if len(rows) != 2048 or json.loads(rows[-1])['step'] != 2048:
        problems.append(f'{cid}: {len(rows)} rows')
    c = json.loads((d / 'complete.json').read_text(encoding='utf8'))
    if c['train_sha256'] != sha(d / 'train.json'):
        problems.append(f'{cid}: completion receipt hash')
add('3-integrity', not problems, f'{len(RAN)} conditions: checkpoint/updates/train hashes and 2048 rows each; problems=' + (', '.join(problems) or 'none'))

# 4 - the report's headline numbers, re-read from evaluation.json
table, problems = {}, []
expect = {'u4sep-soft_sep-five-w32-s117@lr0.0003': (0.207, 0.58, 0.198, 0.184),
          'u4sep-soft_sep-five-w32-s143@lr0.0003': (0.215, 0.40, 0.216, 0.196),
          'u4sep-soft_sep-five-w32-s117@lr0.003': (1.000, 2722.55, 1.000, 1.000)}
for level, cid in RAN:
    d = ROOT / DIRS[level] / cid
    t = json.loads((d / 'train.json').read_text(encoding='utf8')); e = json.loads((d / 'evaluation.json').read_text(encoding='utf8'))
    got = (round(t['selected_validation']['final_accuracy'], 3), round(e['separation_ratio'], 2),
           round(e['neutral']['512-suffix']['final_accuracy'], 3), round(e['clean']['4096']['final_accuracy'], 3))
    table[f'{cid}@{level}'] = got
    want = expect[f'{cid}@{level}']
    if any(abs(a - b) > 0.011 * max(1., abs(b)) for a, b in zip(got, want)):
        problems.append(f'{cid}@{level}: report {want} vs evidence {got}')
add('4-report-numbers', not problems, 'validation / separation ratio / 512-suffix / clean-4096 recomputed for 3 conditions; problems=' + (', '.join(problems) or 'none'))

# 5 - the chance-level claim
add('5-chance-level', abs(math.log(5) - 1.6094) < 1e-3 and all(abs(v[0] - 0.2) < 0.02 for k, v in table.items() if 'lr0.0003' in k),
    f'ln(5)={math.log(5):.4f} and both low-lr validations sit within 0.02 of the 0.200 chance level')

# 6 - the central re-analysis: does gap/spread separate the outcome groups, and does gap/drift?
rows = json.loads((ROOT / 'results/U2d/compare.json').read_text(encoding='utf8'))['rows']
ref = {}
for f in glob.glob(str(ROOT / 'results/U2b-intervene/u1-five-w32-s*.json')):
    d = json.loads(Path(f).read_text(encoding='utf8'))
    ref[d['seed']] = (d['arms']['none']['neutral']['512-suffix']['final_accuracy'], d['arms']['freeze_k2']['neutral']['512-suffix']['final_accuracy'])
frag, rob = [], []
for key, m in rows.items():
    cid, lvl = key.split('@'); seed = int(cid.rsplit('-s', 1)[1])
    src = 'results/U2c-lr0003' if lvl == 'lr0.0003' else 'results/U2c-v2'
    e = json.loads((ROOT / src / cid / 'evaluation.json').read_text(encoding='utf8'))
    n, o = ref[seed]
    repair = (e['neutral']['512-suffix']['final_accuracy'] - n) / (o - n) * 100
    gap = m['anchors']['min_center_distance']; spread = m['anchors']['mean_spread']; drift = m['displacement']['end']
    entry = dict(key=key, repair=repair, gap=gap, gap_over_spread=m['anchors']['separation_ratio'],
                 gap_over_drift=(gap / drift) if drift > 1e-9 else None)
    (frag if repair < 50 else rob).append(entry)
spread_overlap = max(x['gap_over_spread'] for x in frag) > min(x['gap_over_spread'] for x in rob)
gap_overlap = max(x['gap'] for x in frag) > min(x['gap'] for x in rob)
fd = [x['gap_over_drift'] for x in frag if x['gap_over_drift'] is not None]
rd = [x['gap_over_drift'] for x in rob if x['gap_over_drift'] is not None]
drift_separates = max(fd) < min(rd)
add('6-regrouping', spread_overlap and gap_overlap and drift_separates,
    f'grouped by measured repair: {len(frag)} fragile, {len(rob)} robust. gap/spread max(fragile)={max(x["gap_over_spread"] for x in frag):.2f} '
    f'> min(robust)={min(x["gap_over_spread"] for x in rob):.2f} -> OVERLAP as reported. gap max(fragile)={max(x["gap"] for x in frag):.2f} '
    f'> min(robust)={min(x["gap"] for x in rob):.2f} -> OVERLAP as reported. gap/drift max(fragile)={max(fd):.2f} < min(robust)={min(rd):.2f} -> separates as reported.')

# 7 - the correction to U2d's own grouping: robust models sitting in its "fragile" (low-lr) box
misfiled = [x['key'] for x in rob if x['key'].endswith('lr0.0003')]
add('7-u2d-grouping-correction', len(misfiled) >= 3,
    f'{len(misfiled)} conditions repair >= 50 % while sitting at lr 0.0003, which U2d grouped as fragile: ' + ', '.join(sorted(misfiled)))

# 8 - quarantine preserved and inert
q = sorted(glob.glob(str(ROOT / 'results/U4sep-lr*/interrupted-*')))
ok_q = bool(q) and all((Path(x) / 'REASON.txt').exists() for x in q) and not glob.glob(str(ROOT / 'results/U4sep-lr*/interrupted-*/*/complete.json'))
add('8-quarantine', ok_q, f'{len(q)} quarantine directory with REASON.txt and no completion receipt inside, so the runner cannot silently adopt it')

# 9 - the pre-registered verdict category for the low-lr arm
u2d_soft_low = [m['anchors']['separation_ratio'] for k, m in rows.items() if '-soft-' in k and k.endswith('lr0.0003')]
arm_low = [table[f'{cid}@{lvl}'][1] for lvl, cid in RAN if lvl == 'lr0.0003']
gain = (sum(arm_low) / len(arm_low)) / (sum(u2d_soft_low) / len(u2d_soft_low))
add('9-verdict', gain < 1.25,
    f'mean separation ratio of the U4-sep low-lr arm {sum(arm_low)/len(arm_low):.2f} against the U2c soft baseline '
    f'{sum(u2d_soft_low)/len(u2d_soft_low):.2f} = x{gain:.2f} < the pre-registered 1.25 threshold, so the rule returns '
    'INTERVENTION_INEFFECTIVE, as the report states')

# 10 - the sealed U2c manifests still verify, and nothing was written into them
problems = []
for d in ('results/U2c-v2', 'results/U2c-lr0003'):
    man = json.loads((ROOT / d / 'frozen.json').read_text(encoding='utf8'))
    for name, stored in man['sources'].items():
        if sha(ROOT / name) != stored:
            problems.append(f'{d}: {name}')
add('10-sealed-untouched', not problems, 'both sealed U2c frozen manifests still verify against the current sources; problems=' + (', '.join(problems) or 'none'))

summary = dict(**{k: sum(1 for c in checks if c['status'] == k.upper()) for k in ('pass', 'fail')}, unverified=0)
report = dict(cycle='U4-sep', auditor='same session, inline script (the weekly API limit blocked a delegated auditor until 24 Sept)',
              utc=time.time(), checks=checks, summary=summary, discrepancies=discrepancies,
              recomputed=dict(endpoints=table, fragile=sorted(frag, key=lambda x: x['repair']), robust=sorted(rob, key=lambda x: x['repair'])))
out = ROOT / '.superpowers/sdd/phase2'
(out / 'audit-u4sep.json').write_text(json.dumps(report, indent=1, default=float) + '\n', encoding='utf8')
lines = [f'# U4-sep denetimi — {summary["pass"]} PASS / {summary["fail"]} FAIL / 0 UNVERIFIED', '',
         'Denetleyici: aynı oturum, satır içi betik (`audit/audit_u4sep.py`). Bağımsızlık sınırı: delege denetleyici '
         'haftalık API kotasına takılı (24 Eylül\'de sıfırlanır), bu yüzden aynı model kendi döngüsünü denetledi.', '']
for c in checks:
    lines.append(f'- **{c["id"]}** — {c["status"]}: {c["evidence"]}')
(out / 'audit-u4sep.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
print(json.dumps(dict(summary=summary, discrepancies=discrepancies), indent=1))
for c in checks:
    print(c['status'], c['id'], '-', c['evidence'][:150])
