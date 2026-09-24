"""Inline audit of the U4-sep2 cycle: recompute every claim in notes/U4sep2_RAPORU.md from raw evidence.

Includes the two claims the report makes about itself - that the paired design is intact (both arms of a pair start
from the identical frozen checkpoint) and that the supervised controls are not null controls (their U2c protocol
carries a gate-supervision term that the fine-tuner drops) - and the retraction of the endpoint-asymmetry claim.

Writes .superpowers/sdd/phase2/audit-u4sep2.{json,md}. Read-only with respect to results/.
"""
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizon import gated_sweep as gs
from horizon import sep_finetune as sf
from horizon import sweep as sw
from horizon import u4sep2_analyze as ua

OUT = ROOT / 'results/U4sep2'
CK = [('soft', 117), ('soft', 143), ('supervised', 117), ('supervised', 143), ('supervised', 191)]
checks, discrepancies = [], []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add(cid, ok, evidence):
    checks.append(dict(id=cid, status='PASS' if ok else 'FAIL', evidence=evidence))
    if not ok:
        discrepancies.append(cid + ': ' + evidence)


# 1 - freeze integrity
problems = []
if sha(OUT / 'frozen.json') != (OUT / 'frozen.sha256').read_text(encoding='ascii').strip():
    problems.append('sidecar mismatch')
man = json.loads((OUT / 'frozen.json').read_text(encoding='utf8'))
if man['protocol_hash'] != sw.digest(man['protocol']):
    problems.append('protocol_hash does not match its protocol')
for name, stored in man['sources'].items():
    if sha(ROOT / name) != stored:
        problems.append('source drift ' + name)
add('1-freeze', not problems, f'{len(man["sources"])} sources, protocol_hash {man["protocol_hash"][:16]}; problems=' + (', '.join(problems) or 'none'))

# 2 - completeness
complete = sorted(Path(p).parent.name for p in glob.glob(str(OUT / 'u4sep2-*/complete.json')))
evaluated = sorted(Path(p).parent.name for p in glob.glob(str(OUT / 'u4sep2-*/evaluation.json')))
add('2-completeness', len(complete) == 10 and len(evaluated) == 10, f'{len(complete)}/10 complete, {len(evaluated)}/10 evaluated')

# 3 - per-condition integrity
problems = []
for cid in complete:
    d = OUT / cid
    t = json.loads((d / 'train.json').read_text(encoding='utf8'))
    if sha(d / 'selected.pt') != t['selected_sha256'] or sha(d / 'updates.jsonl') != t['updates_sha256']:
        problems.append(cid + ': hash')
    rows = (d / 'updates.jsonl').read_text(encoding='utf8').strip().splitlines()
    if len(rows) != man['protocol']['steps'] or json.loads(rows[-1])['step'] != man['protocol']['steps']:
        problems.append(f'{cid}: {len(rows)} rows')
    if json.loads((d / 'complete.json').read_text(encoding='utf8'))['train_sha256'] != sha(d / 'train.json'):
        problems.append(cid + ': receipt')
add('3-integrity', not problems, f'{len(complete)} conditions: checkpoint/updates/train hashes and {man["protocol"]["steps"]} rows each; problems=' + (', '.join(problems) or 'none'))

# 4 - the paired design: both arms of a pair start from the SAME frozen checkpoint
problems = []
for arm_name, seed in CK:
    src = ROOT / 'results/U2c-lr0003' / f'u2c-{arm_name}-five-w32-s{seed}' / 'selected.pt'
    want = sha(src)
    for arm in ('none', 'hinge'):
        t = json.loads((OUT / f'u4sep2-{arm}-from_{arm_name}-s{seed}' / 'train.json').read_text(encoding='utf8'))
        if t['source_sha256'] != want:
            problems.append(f'{arm_name}/s{seed}/{arm}')
add('4-pairing', not problems, 'both arms of all 5 pairs load the identical frozen U2c checkpoint (sha compared); problems=' + (', '.join(problems) or 'none'))

# 5 - the design flaw claim: supervised checkpoints were trained WITH a gate-supervision term
u2c = gs.verify_frozen('results/U2c-lr0003')
weights = {arm: gs.gate_weight(arm, u2c['protocol']) for arm in ('soft', 'supervised')}
add('5-control-not-null', weights['supervised'] > 0 and weights['soft'] == 0,
    f'U2c gate_loss weight: soft={weights["soft"]}, supervised={weights["supervised"]}. The fine-tuner uses neither, so the '
    'control is a true null only for the soft checkpoints - exactly as the report states.')

# 6 - the headline pair table, recomputed
manifest, pairs, rows = ua.collect(OUT)
expect = {'soft/s117': (-40.7, -40.7, 54.4), 'soft/s143': (16.6, 27.5, 52.4), 'supervised/s117': (-40.7, 61.8, -44.2),
          'supervised/s143': (-30.7, 87.5, 84.7), 'supervised/s191': (-99.5, -51.8, -103.6)}
problems = []
for key, (u, c, h) in expect.items():
    e = rows[key]
    got = (e['u2c_repair'], e['none_repair'], e['hinge_repair'])
    if any(abs(a - b) > 0.15 for a, b in zip(got, (u, c, h))):
        problems.append(f'{key}: report {(u, c, h)} vs evidence {tuple(round(x, 1) for x in got)}')
add('6-pair-table', not problems, '5 pairs: U2c / control / hinge repair recomputed from the frozen U2b reference; problems=' + (', '.join(problems) or 'none'))

# 7 - the sign test
scored = [dict(p, control_repair=p['none_repair'], validation_control=p['validation_none'], ratio_control=p['ratio_none']) for p in pairs]
st = ua.sign_test(scored)
add('7-sign-test', st['ratio_hits'] == 2 and st['drift_hits'] == 4,
    f'over {st["pairs"]} matched pairs: gap/drift ratio predicts the repair direction {st["ratio_hits"]}/5, '
    f'absolute drift (inverted) {st["drift_hits"]}/5 - as reported')

# 8 - the ratio rose in every pair, which is why it cannot explain the direction
rose = [p['checkpoint'] for p in scored if (p['eval_ratio_hinge'] or 0) > (p['eval_ratio_none'] or 0)]
target = ua.THRESHOLDS['ratio_target']
reached_eval = [p['checkpoint'] for p in scored if (p['eval_ratio_hinge'] or 0) >= target]
reached_train = [p['checkpoint'] for p in scored if (p['ratio_hinge'] or 0) >= target]
# Previously this asserted "the intervention worked in all five" from a mere RISE. An independent review showed that
# conflates two different things: the ratio rising above its control, and the ratio reaching the pre-registered
# target. They disagree, because the rule reads a 64-slot training measurement while the target was calibrated on the
# 512-slot evaluation measurement. Both are now counted and the discrepancy is part of the evidence.
add('8-ratio-rose-and-target', len(rose) == 5 and len(reached_train) == 5 and len(reached_eval) == 1,
    f'{len(rose)}/5 pairs raised gap/drift above their paired control - the intervention moved the quantity in every '
    f'pair. But the pre-registered target of {target} is reached by {len(reached_train)}/5 in the TRAINING measurement '
    f'(64 neutral slots, what the rule reads) and only {len(reached_eval)}/5 in the EVALUATION measurement (512 slots, '
    f'what the target was calibrated on: fragile <= 0.57, robust >= 0.92). Scored with the evaluation ratio the verdict '
    f'becomes INTERVENTION_INEFFECTIVE, not CONFOUNDED_BY_TRAINING. Recorded in the report as correction 7.')

# 9 - the retraction: the endpoint asymmetry does not generalise
keys = ['128-prefix', '128-middle', '128-suffix', '512-prefix', '512-middle', '512-suffix']
avg = {}
for k in keys:
    diffs = []
    for arm_name, seed in CK:
        c = json.loads((OUT / f'u4sep2-none-from_{arm_name}-s{seed}' / 'evaluation.json').read_text(encoding='utf8'))
        h = json.loads((OUT / f'u4sep2-hinge-from_{arm_name}-s{seed}' / 'evaluation.json').read_text(encoding='utf8'))
        diffs.append(h['neutral'][k]['final_accuracy'] - c['neutral'][k]['final_accuracy'])
    avg[k] = sum(diffs) / len(diffs)
pre = (avg['128-prefix'] + avg['512-prefix']) / 2
suf = (avg['128-suffix'] + avg['512-suffix']) / 2
add('9-asymmetry-retracted', pre > 0 > suf,
    f'mean hinge-minus-control by placement: prefix {pre:+.3f}, middle {(avg["128-middle"] + avg["512-middle"]) / 2:+.3f}, suffix {suf:+.3f}. '
    'The suffix gain seen in the two soft pairs reverses overall, so the earlier "the asymmetry replicates" claim is correctly retracted.')

# 10 - the pre-registered verdict, recomputed, and the sealed U2c sources
v = ua.verdict(scored)
problems = [name for name, stored in json.loads((ROOT / 'results/U2c-lr0003/frozen.json').read_text(encoding='utf8'))['sources'].items()
            if sha(ROOT / name) != stored]
add('10-verdict-and-seals', v['status'] == 'CONFOUNDED_BY_TRAINING' and v['pairs_rescued'] == 2 and not problems,
    f'rule returns {v["status"]} with {v["pairs_rescued"]}/5 rescued, which is what the report prints; '
    'sealed U2c-lr0003 sources still verify; problems=' + (', '.join(problems) or 'none'))

summary = dict(**{k: sum(1 for c in checks if c['status'] == k.upper()) for k in ('pass', 'fail')}, unverified=0)
report = dict(cycle='U4-sep2', auditor='same session, inline script (the weekly API limit blocked a delegated auditor until 24 Sept)',
              utc=time.time(), checks=checks, summary=summary, discrepancies=discrepancies,
              recomputed=dict(pairs={k: {kk: vv for kk, vv in r.items() if not kk.startswith('endpoints')} for k, r in rows.items()},
                              sign_test=st, verdict=v, placement_means=avg))
out = ROOT / '.superpowers/sdd/phase2'
(out / 'audit-u4sep2.json').write_text(json.dumps(report, indent=1, default=float) + '\n', encoding='utf8')
lines = [f'# U4-sep2 denetimi — {summary["pass"]} PASS / {summary["fail"]} FAIL / 0 UNVERIFIED', '',
         'Denetleyici: aynı oturum, satır içi betik (`audit/audit_u4sep2.py`). Bağımsızlık sınırı: delege denetleyici '
         'haftalık API kotasına takılı (24 Eylül\'de sıfırlanır).', '']
for c in checks:
    lines.append(f'- **{c["id"]}** — {c["status"]}: {c["evidence"]}')
(out / 'audit-u4sep2.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
print(json.dumps(dict(summary=summary, discrepancies=discrepancies), indent=1))
for c in checks:
    print(c['status'], c['id'], '-', c['evidence'][:140])
