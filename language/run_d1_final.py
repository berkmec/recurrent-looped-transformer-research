"""D1 final with batch-level resume (package E7), created after the 2026-09-18 power outage killed the 17-hour run.

Same protocol values, sampling, scoring, gates and outputs as language.run_d1 (protocol version 2 only adds the resume
contract), but every generated batch is appended to `final-{arm}-progress.jsonl` (fsync) keyed by the protocol hash, and a
restart skips the finished example ids. Rows carry `resumed` so the report can state how many predictions came from a
previous attempt. A completion receipt (E12) is written next to final.json.
CLI: --freeze [--protocol-json '{...}'] | --final   (--out results/D1-final)
"""
import argparse
import copy
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

from audit import receipt as rc
from language import dataset as ds
from language import extract as ex
from language import run_d1 as rd
from language import score_semantics as sc

ROOT = rd.ROOT
SOURCES = rd.SOURCES + ('language/run_d1_final.py', 'audit/receipt.py')


def protocol():
    p = rd.protocol()
    p.update(version=2, resume='batch-level progress JSONL per arm keyed by protocol hash; restart skips finished ids; rows marked resumed',
             purpose=p['purpose'] + ' — final run with resumable extraction (after the 2026-09-18 power outage)',
             development_reference='results/D1 (dev32 complete; dev200 INCOMPLETE: LM arms lost in the outage)')
    return p


def source_hashes():
    return {name: rd.sha(ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol())
    manifest = dict(protocol=p, protocol_hash=rd.digest(p), sources=source_hashes(), environment=dict(python=platform.python_version()), frozen_utc=time.time())
    if (out / 'frozen.json').exists():
        old = verify_frozen(out)
        if old['protocol_hash'] != manifest['protocol_hash']:
            raise ValueError('refusing frozen protocol change')
        return old
    if any(out.iterdir()):
        raise ValueError('new freeze requires empty output directory')
    rd.write_json(out / 'frozen.json', manifest)
    (out / 'frozen.sha256').write_text(rd.sha(out / 'frozen.json'), encoding='ascii')
    return manifest


def verify_frozen(out):
    out = Path(out)
    if rd.sha(out / 'frozen.json') != (out / 'frozen.sha256').read_text(encoding='ascii').strip():
        raise ValueError('frozen manifest hash mismatch')
    manifest = json.loads((out / 'frozen.json').read_text(encoding='utf8'))
    if manifest['protocol_hash'] != rd.digest(manifest['protocol']) or manifest['sources'] != source_hashes():
        raise ValueError('protocol/source drift')
    return manifest


def extract_resumable(examples, generator, arm, batch_size, max_new_tokens, progress_path, run_key):
    if arm == 'finite_parser':
        return [dict(ex.finite_parser(e), resumed=False) for e in examples]
    if arm not in ('lm_zero', 'lm_schema'):
        raise ValueError('unknown arm')
    progress_path = Path(progress_path); done = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding='utf8').splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get('run_key') == run_key and row.get('arm') == arm:
                    done[row['id']] = row['pred']
    prompts = {e['id']: ex.build_prompt(e, arm) for e in examples}
    todo = [e for e in examples if e['id'] not in done]
    fresh = {}
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open('a', encoding='utf8') as f:
        for at in range(0, len(todo), batch_size):
            chunk = todo[at:at + batch_size]
            outputs = generator.generate([prompts[e['id']] for e in chunk], max_new_tokens=max_new_tokens)
            for o, e in zip(outputs, chunk):
                pred = dict(ex.parse_output(o, {c['pid'] for c in e['context']}), prompt_chars=len(prompts[e['id']]))
                fresh[e['id']] = pred
                f.write(json.dumps(dict(run_key=run_key, arm=arm, id=e['id'], pred=pred, utc=time.time()), ensure_ascii=False) + '\n')
            f.flush(); os.fsync(f.fileno())
    return [dict(done[e['id']], resumed=True) if e['id'] in done else dict(fresh[e['id']], resumed=False) for e in examples]


def run_arm(arm, examples, controls, generator, p, out, tag, run_key):
    t0 = time.perf_counter(); out = Path(out)
    preds = extract_resumable(examples, generator, arm, p['batch_size'], p['max_new_tokens'], out / f'{tag}-{arm}-progress.jsonl', run_key)
    scores = [sc.score_example(pr, e['gold']) for pr, e in zip(preds, examples)]
    agg = sc.aggregate(scores)
    groups = [e['group'] for e in examples]
    f1_per = [2 * s['relation_tp'] / (2 * s['relation_tp'] + s['relation_fp'] + s['relation_fn']) if (2 * s['relation_tp'] + s['relation_fp'] + s['relation_fn']) else 0. for s in scores]
    agg['relation_f1_group_ci95'] = rd.group_bootstrap(f1_per, groups, p['bootstrap'], p['seed'] + 7)
    agg['by_hops'] = {}
    for k in sorted({e['hops'] for e in examples}):
        sub = [s for s, e in zip(scores, examples) if e['hops'] == k]
        agg['by_hops'][str(k)] = dict(examples=len(sub), relation_f1=sc.aggregate(sub)['relation_f1'], coverage=sc.aggregate(sub)['coverage'])
    control_result = None
    if controls:
        cpreds = extract_resumable(controls, generator, arm, p['batch_size'], p['max_new_tokens'], out / f'{tag}-{arm}-controls-progress.jsonl', run_key)
        cscores = [sc.score_example(pr, c['gold'], expect_unresolved=c['expect_unresolved_hops']) for pr, c in zip(cpreds, controls)]
        by_kind = {}
        for kind in sorted({c['control'] for c in controls}):
            sub = [s for s, c in zip(cscores, controls) if c['control'] == kind]
            n_exp = sum(s['controls_expected'] for s in sub); ok = sum(s['controls_ok'] for s in sub)
            wrong = sum(min(s['accepted_wrong'], s['controls_expected']) for s in sub)
            by_kind[kind] = dict(controls=len(sub), expected_unresolved=n_exp, correctly_unresolved=ok, false_accepts=wrong, false_accept_rate=wrong / n_exp if n_exp else None)
        control_result = dict(by_kind=by_kind, total_false_accepts=sum(v['false_accepts'] for v in by_kind.values()), total_expected=sum(v['expected_unresolved'] for v in by_kind.values()),
                              resumed_rows=sum(1 for pr in cpreds if pr.get('resumed')))
        rd.write_json(out / f'{tag}-{arm}-controls.json', dict(arm=arm, rows=[dict(id=c['id'], control=c['control'], expect=c['expect_unresolved_hops'], pred=pr, score=s) for c, pr, s in zip(controls, cpreds, cscores)]))
    raw = out / f'{tag}-{arm}.json'
    rd.write_json(raw, dict(arm=arm, rows=[dict(id=e['id'], hops=e['hops'], gold=e['gold'], pred=pr, score=s) for e, pr, s in zip(examples, preds, scores)]))
    result = dict(arm=arm, metrics=agg, errors=rd.classify_errors(preds, examples), controls=control_result, raw_file=raw.name, raw_sha256=rd.sha(raw),
                  resumed_rows=sum(1 for pr in preds if pr.get('resumed')), seconds=time.perf_counter() - t0,
                  generator=getattr(generator, 'stats', lambda: None)() if generator is not None else None)
    if arm != 'finite_parser':
        t = p['targets']
        result['gates'] = dict(relation_f1=agg['relation_f1'] >= t['relation_f1'], risk=(agg['risk_upper_95'] is not None and agg['risk_upper_95'] <= t['risk_upper']), coverage=agg['coverage'] >= t['coverage'])
        result['status'] = 'MEASURED_POSITIVE' if all(result['gates'].values()) else 'MEASURED_NEGATIVE'
    return result


def run_final(out, generator=None):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    if (out / 'final.json').exists():
        return json.loads((out / 'final.json').read_text(encoding='utf8'))
    run_key = manifest['protocol_hash']
    examples = rd.sample_examples(p['final_path'], p['final_n'], p['distractors'], p['seed'])
    controls = ds.negative_controls(examples[:p['controls_n']], seed=p['seed'])
    generator = generator or rd.make_generator(p)
    arms = {arm: run_arm(arm, examples, controls, generator if arm != 'finite_parser' else None, p, out, 'final', run_key) for arm in p['arms']}
    result = dict(status='MEASURED', protocol_hash=manifest['protocol_hash'], data=dict(path=p['final_path'], sha256=ds.sha256_file(p['final_path']), examples=len(examples),
                  hops=dict(Counter(e['hops'] for e in examples)), controls=len(controls), controls_by_kind=dict(Counter(c['control'] for c in controls))),
                  arms=arms, targets=p['targets'], development_reference=p['development_reference'], turkish_slice='UNMEASURED',
                  resumed_rows={arm: r['resumed_rows'] for arm, r in arms.items()})
    rd.write_json(out / 'final.json', result)
    artifacts = {'final': out / 'final.json', **{f'raw-{arm}': out / r['raw_file'] for arm, r in arms.items()}}
    rc.write_receipt(out / 'receipt.json', rc.make_receipt(f'd1-final-{run_key[:12]}', artifacts, returncode=0, expected_items=len(p['arms']), completed_items=len(arms), tests_passed=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--final', action='store_true')
    parser.add_argument('--out', default='results/D1-final'); parser.add_argument('--protocol-json')
    args = parser.parse_args()
    overrides = json.loads(args.protocol_json) if args.protocol_json else {}
    if args.freeze:
        result = dict(protocol_hash=freeze(args.out, dict(protocol(), **overrides))['protocol_hash'])
    else:
        r = run_final(args.out)
        result = {arm: dict(status=v.get('status'), f1=v['metrics']['relation_f1'], coverage=v['metrics']['coverage'], risk=v['metrics']['risk_upper_95'], resumed=v['resumed_rows']) for arm, v in r['arms'].items()}
    print(json.dumps(result, indent=1, allow_nan=False, default=str, ensure_ascii=False))


if __name__ == '__main__':
    main()
