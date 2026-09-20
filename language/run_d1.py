"""D1 runner: freeze, development diagnostics (32 then 200), final with negative controls, group bootstrap, target gates."""
import argparse
import copy
import hashlib
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from language import dataset as ds
from language import extract as ex
from language import score_semantics as sc

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('language/dataset.py', 'language/extract.py', 'language/score_semantics.py', 'language/generator.py', 'language/run_d1.py', 'memory/api.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=False, ensure_ascii=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def protocol():
    return dict(version=1, purpose='D1 free text -> headings, claims, source bindings on MuSiQue decompositions',
                development_path='E:/rlt-research/datasets/dgslibisey__MuSiQue/musique_ans_v1.0_train.jsonl', development_stages=[32, 200],
                final_path='E:/rlt-research/datasets/dgslibisey__MuSiQue/musique_ans_v1.0_dev.jsonl', final_n=600, controls_n=100,
                distractors=2, seed=2026091810, batch_size=4, max_new_tokens=384,
                model_dir='E:/rlt-research/models/Qwen3-4B', quant='nf4', decoding='greedy, thinking disabled',
                arms=['finite_parser', 'lm_zero', 'lm_schema'], bootstrap=1000,
                targets=dict(relation_f1=.95, risk_upper=.01, coverage=.80),
                note='human-authored gold (MuSiQue decomposition); controls are programmatic perturbations; Turkish UNMEASURED')


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol())
    manifest = dict(protocol=p, protocol_hash=digest(p), sources=source_hashes(), environment=dict(python=platform.python_version()), frozen_utc=time.time())
    if (out / 'frozen.json').exists():
        old = verify_frozen(out)
        if old['protocol_hash'] != manifest['protocol_hash']:
            raise ValueError('refusing frozen protocol change')
        return old
    if any(out.iterdir()):
        raise ValueError('new freeze requires empty output directory')
    write_json(out / 'frozen.json', manifest)
    (out / 'frozen.sha256').write_text(sha(out / 'frozen.json'), encoding='ascii')
    return manifest


def verify_frozen(out):
    out = Path(out)
    if sha(out / 'frozen.json') != (out / 'frozen.sha256').read_text(encoding='ascii').strip():
        raise ValueError('frozen manifest hash mismatch')
    manifest = json.loads((out / 'frozen.json').read_text(encoding='utf8'))
    if manifest['protocol_hash'] != digest(manifest['protocol']) or manifest['sources'] != source_hashes():
        raise ValueError('protocol/source drift')
    return manifest


def sample_examples(path, n, distractors, seed):
    everything = ds.build_examples(path, distractors=distractors, seed=seed)
    rng = np.random.default_rng(seed + 1)
    if n is None or n >= len(everything):
        return everything
    pick = sorted(rng.choice(len(everything), size=n, replace=False))
    return [everything[i] for i in pick]


def make_generator(p):
    from language.generator import QwenGenerator
    return QwenGenerator(p['model_dir'], quant=p['quant'])


def group_bootstrap(values, groups, n, seed):
    values = np.asarray(values, dtype=np.float64); groups = np.asarray(groups)
    unique = np.unique(groups); per = np.array([values[groups == g].mean() for g in unique])
    rng = np.random.default_rng(seed)
    means = [per[rng.integers(0, len(unique), len(unique))].mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def classify_errors(preds, examples):
    kinds = Counter()
    for pr, e in zip(preds, examples):
        if pr.get('parse') != 'OK':
            kinds[pr.get('parse', 'PARSE')] += 1; continue
        gold = {h['step']: h for h in e['gold']['hops']}
        for step, g in gold.items():
            p = next((h for h in pr['hops'] if h.get('step') == step), None)
            if p is None or p.get('unresolved'):
                kinds['unresolved'] += 1
            elif not sc.relation_match(p.get('relation'), g['relation']):
                kinds['wrong_relation'] += 1
            elif not sc.answer_match(p.get('answer'), g['answer'], g.get('aliases', ())):
                kinds['wrong_answer'] += 1
            elif p.get('source') != g['source']:
                kinds['wrong_source'] += 1
            else:
                kinds['correct'] += 1
        kinds['extra_hops'] += sum(1 for h in pr['hops'] if h.get('step') not in gold)
    return dict(kinds)


def run_arm(arm, examples, controls, generator, p, out, tag):
    t0 = time.perf_counter()
    preds = ex.extract(examples, generator, arm=arm, batch_size=p['batch_size'], max_new_tokens=p['max_new_tokens'])
    scores = [sc.score_example(pr, e['gold']) for pr, e in zip(preds, examples)]
    agg = sc.aggregate(scores)
    groups = [e['group'] for e in examples]
    f1_per = [2 * s['relation_tp'] / (2 * s['relation_tp'] + s['relation_fp'] + s['relation_fn']) if (2 * s['relation_tp'] + s['relation_fp'] + s['relation_fn']) else 0. for s in scores]
    agg['relation_f1_group_ci95'] = group_bootstrap(f1_per, groups, p['bootstrap'], p['seed'] + 7)
    agg['by_hops'] = {}
    for k in sorted({e['hops'] for e in examples}):
        sub = [s for s, e in zip(scores, examples) if e['hops'] == k]
        agg['by_hops'][str(k)] = dict(examples=len(sub), relation_f1=sc.aggregate(sub)['relation_f1'], coverage=sc.aggregate(sub)['coverage'])
    control_result = None
    if controls:
        cpreds = ex.extract(controls, generator, arm=arm, batch_size=p['batch_size'], max_new_tokens=p['max_new_tokens'])
        cscores = [sc.score_example(pr, c['gold'], expect_unresolved=c['expect_unresolved_hops']) for pr, c in zip(cpreds, controls)]
        by_kind = {}
        for kind in sorted({c['control'] for c in controls}):
            sub = [s for s, c in zip(cscores, controls) if c['control'] == kind]
            n_exp = sum(s['controls_expected'] for s in sub); ok = sum(s['controls_ok'] for s in sub)
            wrong_on_perturbed = sum(min(s['accepted_wrong'], s['controls_expected']) for s in sub)
            by_kind[kind] = dict(controls=len(sub), expected_unresolved=n_exp, correctly_unresolved=ok, false_accepts=wrong_on_perturbed,
                                 false_accept_rate=wrong_on_perturbed / n_exp if n_exp else None)
        control_result = dict(by_kind=by_kind, total_false_accepts=sum(v['false_accepts'] for v in by_kind.values()), total_expected=sum(v['expected_unresolved'] for v in by_kind.values()))
        write_json(Path(out) / f'{tag}-{arm}-controls.json', dict(arm=arm, rows=[dict(id=c['id'], control=c['control'], expect=c['expect_unresolved_hops'], pred=pr, score=s) for c, pr, s in zip(controls, cpreds, cscores)]))
    raw = Path(out) / f'{tag}-{arm}.json'
    write_json(raw, dict(arm=arm, rows=[dict(id=e['id'], hops=e['hops'], gold=e['gold'], pred=pr, score=s) for e, pr, s in zip(examples, preds, scores)]))
    result = dict(arm=arm, metrics=agg, errors=classify_errors(preds, examples), controls=control_result, raw_file=raw.name, raw_sha256=sha(raw),
                  seconds=time.perf_counter() - t0, generator=getattr(generator, 'stats', lambda: None)() if generator is not None else None)
    if arm != 'finite_parser':
        t = p['targets']
        result['gates'] = dict(relation_f1=agg['relation_f1'] >= t['relation_f1'], risk=(agg['risk_upper_95'] is not None and agg['risk_upper_95'] <= t['risk_upper']),
                               coverage=agg['coverage'] >= t['coverage'])
        result['status'] = 'MEASURED_POSITIVE' if all(result['gates'].values()) else 'MEASURED_NEGATIVE'
    return result


def run_development(out, generator=None):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    generator = generator or make_generator(p)
    stages = {}
    for n in p['development_stages']:
        examples = sample_examples(p['development_path'], n, p['distractors'], p['seed'] + n)
        controls = ds.negative_controls(examples[:min(len(examples), max(4, n // 5))], seed=p['seed'])
        stages[str(n)] = {arm: run_arm(arm, examples, controls, generator if arm != 'finite_parser' else None, p, out, f'dev{n}') for arm in p['arms']}
        write_json(out / 'development.json', dict(stages=stages, role='development diagnostics only; not final evidence'))
    return stages


def run_final(out, generator=None):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    if (out / 'final.json').exists():
        return json.loads((out / 'final.json').read_text(encoding='utf8'))
    examples = sample_examples(p['final_path'], p['final_n'], p['distractors'], p['seed'])
    controls = ds.negative_controls(examples[:p['controls_n']], seed=p['seed'])
    generator = generator or make_generator(p)
    arms = {arm: run_arm(arm, examples, controls, generator if arm != 'finite_parser' else None, p, out, 'final') for arm in p['arms']}
    result = dict(status='MEASURED', protocol_hash=manifest['protocol_hash'], data=dict(path=p['final_path'], sha256=ds.sha256_file(p['final_path']), examples=len(examples),
                  hops=dict(Counter(e['hops'] for e in examples)), controls=len(controls), controls_by_kind=dict(Counter(c['control'] for c in controls))),
                  arms=arms, targets=p['targets'], turkish_slice='UNMEASURED')
    write_json(out / 'final.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--development', action='store_true'); group.add_argument('--final', action='store_true')
    parser.add_argument('--out', required=True); parser.add_argument('--protocol-json')
    args = parser.parse_args()
    overrides = json.loads(args.protocol_json) if args.protocol_json else {}
    if args.freeze:
        result = dict(protocol_hash=freeze(args.out, dict(protocol(), **overrides))['protocol_hash'])
    elif args.development:
        stages = run_development(args.out)
        result = {n: {arm: dict(f1=r['metrics']['relation_f1'], coverage=r['metrics']['coverage'], wrong=r['metrics']['accepted_wrong'], errors=r['errors']) for arm, r in s.items()} for n, s in stages.items()}
    else:
        r = run_final(args.out)
        result = {arm: dict(status=v.get('status'), f1=v['metrics']['relation_f1'], coverage=v['metrics']['coverage'], risk=v['metrics']['risk_upper_95']) for arm, v in r['arms'].items()}
    print(json.dumps(result, indent=1, allow_nan=False, default=str, ensure_ascii=False))


if __name__ == '__main__':
    main()
