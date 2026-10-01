"""Frozen fresh R1 evaluation. All oracle use is restricted to scoring."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from learning.api import load_model, final_query_candidates
from memory.api import verify_candidate as old_verify
from .certificate import (verify_path, verify_envelope, bind_candidate,
                          audit_certificate, digest)

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('__RLT_ROOT__/2026-09-16-e2e')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def protocol():
    return dict(version=1, count=512, model_seeds=[17, 43, 91],
                sources=dict(iid=2026091711, depth5=2026091712, depth8=2026091713, branched=2026091714),
                thresholds=[.025, .05, .099999], primary_threshold=.099999,
                correctness_epsilon=.1, max_rounds=8, prior_rounds=6,
                batch=32, lazy_replay_count=16, bootstrap_seed=2026091791,
                bootstrap_repetitions=2000,
                checkpoint='combined-{seed}-step600.pt',
                selection='none; frozen models; no final-data tuning',
                primary='fixed-final-round identical candidates, per model seed and source distribution',
                secondary='first acceptable candidate from identical frozen trace; observed lazy subset separately',
                source_unit='one independent generated problem; rounds and model seeds are dependent repeated measures',
                topology_controls='iid/depth5/depth8 have one anchored chain: component/path sums mathematically equal; branched has four-edge query path and four-edge off-path spur, OOD topology',
                external_truth='conditional on faithful numeric evidence; no natural-language verification claim')


def freeze(out):
    out = Path(out)
    if (out / 'protocol.json').exists():
        return verify_frozen(out)
    p = protocol()
    p['frozen_utc'] = datetime.now(timezone.utc).isoformat()
    p['environment'] = dict(torch=torch.__version__, numpy=np.__version__, device='cpu', threads=1)
    p['source_pins'] = {rel: sha(ROOT / rel) for rel in
                        ['review_cycle/certificate.py', 'review_cycle/certificate_bench.py',
                         'learning/model.py', 'learning/token_core.py', 'learning/api.py', 'memory/api.py']}
    p['checkpoint_pins'] = {str(seed): sha(OLD / 'learning/checkpoints' / p['checkpoint'].format(seed=seed))
                            for seed in p['model_seeds']}
    p['old_manifest_sha256'] = sha(OLD / 'audit/final-manifest.json')
    write(out / 'protocol.json', p)
    (out / 'protocol.sha256').write_text(sha(out / 'protocol.json') + '\n')
    return p


def verify_frozen(out):
    out = Path(out)
    if sha(out / 'protocol.json') != (out / 'protocol.sha256').read_text().strip():
        raise RuntimeError('PROTOCOL_HASH_MISMATCH')
    p = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
    for rel, expected in p['source_pins'].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError('SOURCE_DRIFT: ' + rel)
    for seed, expected in p['checkpoint_pins'].items():
        if sha(OLD / 'learning/checkpoints' / p['checkpoint'].format(seed=seed)) != expected:
            raise RuntimeError('CHECKPOINT_DRIFT')
    if sha(OLD / 'audit/final-manifest.json') != p['old_manifest_sha256']:
        raise RuntimeError('OLD_MANIFEST_DRIFT')
    return p


def verify_receipt(out, p, receipt, split, seed):
    if (receipt['split'] != split or receipt['seed'] != seed
            or receipt['protocol_sha256'] != sha(out / 'protocol.json')
            or receipt['source_sha256'] != sha(out / f'data-{split}.npz')):
        raise RuntimeError('RECEIPT_LINEAGE_MISMATCH')
    expected_files = {f'{kind}-{split}-{seed}.{ext}' for kind, ext in
                      [('raw', 'npz'), ('certificates', 'json'), ('lazy', 'json')]}
    if set(receipt['artifacts']) != expected_files:
        raise RuntimeError('RECEIPT_ARTIFACT_SET_MISMATCH')
    for rel, expected in receipt['artifacts'].items():
        if sha(out / rel) != expected:
            raise RuntimeError('RESULT_HASH_MISMATCH')


def verify_completion(out, p):
    completion = json.loads((out / 'completion.json').read_text())
    if completion['protocol_sha256'] != sha(out / 'protocol.json'):
        raise RuntimeError('COMPLETION_LINEAGE_MISMATCH')
    expected = {(split, seed) for split in p['sources'] for seed in p['model_seeds']}
    observed = [(row['split'], row['seed']) for row in completion['receipts']]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError('COMPLETION_CONDITION_SET_MISMATCH')
    for row in completion['receipts']:
        verify_receipt(out, p, row, row['split'], row['seed'])
    return completion


def make_data(count, seed, split):
    rng = np.random.default_rng(seed)
    rows, flags, inverses, depths = [], [], [], []
    for index in range(count):
        depth = [1, 3][index % 2] if split == 'iid' else dict(depth5=5, depth8=8, branched=4)[split]
        inv, affected = bool((index // 2) % 2), bool((index // 4) % 2)
        ids = rng.permutation(10).tolist()
        chain, other = ids[:depth + 1], ids[depth + 1:]
        parents = {chain[j]: chain[j - 1] for j in range(1, len(chain))}
        if split == 'branched':
            spur, other = ids[5:9], ids[9:]
            parents[spur[0]] = chain[1]
            parents.update({spur[j]: spur[j - 1] for j in range(1, len(spur))})
        else:
            parents.update({other[j]: other[j - 1] for j in range(1, len(other))})
        offsets = {node: float(rng.uniform(-.35, .35)) for node in parents}
        known, requested = (chain[-1], chain[0]) if inv else (chain[0], chain[-1])
        observations = {known: float(rng.uniform(-.7, .7)), other[0]: float(rng.uniform(-.7, .7))}
        facts = [[1, c, p, offsets[c]] for c, p in parents.items()]
        facts += [[0, n, n, value] for n, value in observations.items()]
        rng.shuffle(facts)
        changed = known if affected else other[0]
        replacement = observations[changed] + float(rng.choice([-1, 1]) * rng.uniform(.3, .8))
        query = [2, requested, requested, 0]
        rows.append(facts + [query, [6, 0, 0, 0], [3, changed, changed, replacement], query, [6, 0, 0, 0], query])
        flags.append(affected)
        inverses.append(inv)
        depths.append(depth)
    return dict(events=np.array(rows, np.float32), affected=np.array(flags, bool),
                inverse=np.array(inverses, bool), depth=np.array(depths, np.int64),
                example_id=np.arange(count, dtype=np.int64))


def records(tape):
    result = []
    for op, child, parent, value in tape:
        op, child, parent = int(op), str(int(child)), str(int(parent))
        if op == 6:
            continue
        kind = {0: 'root', 1: 'edge', 2: 'query', 3: 'revision', 4: 'revision'}[op]
        result.append(dict(kind=kind, child=child,
                           parent=parent if op in (1, 4) else None,
                           value=None if op == 2 else float(value)))
    return result


def reference_values(events):
    """Independent exact Gaussian elimination, not certificate graph traversal."""
    latest = {}
    for e in events:
        if e['kind'] in ('query', 'continuation'):
            continue
        parent = e.get('parent')
        latest[(e['child'], parent is None)] = (e['child'], parent, Fraction(e['value']))
    names = sorted({x for child, parent, _ in latest.values() for x in [child, parent] if x is not None})
    pos = {name: i for i, name in enumerate(names)}
    matrix = []
    for child, parent, value in latest.values():
        row = [Fraction(0)] * (len(names) + 1)
        row[pos[child]] += 1
        if parent is not None:
            row[pos[parent]] -= 1
        row[-1] = value
        matrix.append(row)
    pivots, row_index = [], 0
    for col in range(len(names)):
        pivot = next((j for j in range(row_index, len(matrix)) if matrix[j][col]), None)
        if pivot is None:
            continue
        matrix[row_index], matrix[pivot] = matrix[pivot], matrix[row_index]
        divisor = matrix[row_index][col]
        matrix[row_index] = [x / divisor for x in matrix[row_index]]
        for j, row in enumerate(matrix):
            if j != row_index and row[col]:
                factor = row[col]
                matrix[j] = [a - factor * b for a, b in zip(row, matrix[row_index])]
        pivots.append(col)
        row_index += 1
    if any(not any(row[:-1]) and row[-1] for row in matrix):
        raise ValueError('CONTRADICTORY_REFERENCE')
    if len(pivots) != len(names):
        raise ValueError('UNIDENTIFIABLE_REFERENCE')
    return {names[col]: matrix[row][-1] for row, col in enumerate(pivots)}


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    phat, denom = k / n, 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return [max(0., centre - half), min(1., centre + half)]


def metrics(accepted, correct):
    accepted, correct = np.asarray(accepted, bool), np.asarray(correct, bool)
    n, a = len(accepted), int(accepted.sum())
    tp, wrong, total_correct = int((accepted & correct).sum()), int((accepted & ~correct).sum()), int(correct.sum())
    return dict(n=n, accepted=a, correct=total_correct, accepted_correct=tp, accepted_wrong=wrong,
                coverage=a / n, coverage_wilson95=wilson(a, n), accuracy=total_correct / n,
                accuracy_wilson95=wilson(total_correct, n),
                selective_risk=wrong / a if a else None, selective_risk_wilson95=wilson(wrong, a),
                zero_error_upper_one_sided95=1 - .05 ** (1 / a) if a and wrong == 0 else None,
                precision=tp / a if a else None, recall=tp / total_correct if total_correct else None,
                recall_wilson95=wilson(tp, total_correct), correct_rejected=total_correct - tp)


def challenge_checks():
    events = [dict(kind='root', child='a', parent=None, value=0),
              dict(kind='edge', child='q', parent='a', value=1),
              dict(kind='edge', child='b', parent='a', value=2)]
    good = bind_candidate(dict(a=0., q=1., b=99.), events, 'q', 'challenge')
    stale = events + [dict(kind='revision', child='a', parent=None, value=1)]
    contradictory = events + [dict(kind='root', child='b', parent=None, value=3)]
    cases = [
        ('offpath_wrong', good, events, 'q', 'challenge', True),
        ('wrong_query', bind_candidate(dict(a=0., q=1.5, b=2.), events, 'q', 'challenge'), events, 'q', 'challenge', False),
        ('stale', good, stale, 'q', 'challenge', False),
        ('cross_scope', good, events, 'q', 'other', False),
        ('wrong_query_binding', good, events, 'b', 'challenge', False),
        ('corrupt_digest', {**good, 'source_sha256': '0' * 64}, events, 'q', 'challenge', False),
        ('contradiction', bind_candidate(dict(a=0., q=1., b=3.), contradictory, 'q', 'challenge'), contradictory, 'q', 'challenge', False)]
    rows = []
    for title, env, ev, query, scope, accept in cases:
        result = verify_envelope(env, ev, query, scope, .099999)
        passed = (result['status'] == 'VERIFIED') == accept
        rows.append(dict(case=title, expected_accept=accept, result=result, passed=passed))
    if not all(r['passed'] for r in rows):
        raise AssertionError('CHALLENGE_FAILURE')
    return rows


@torch.no_grad()
def traces(model, data, p):
    values = []
    for start in range(0, len(data['events']), p['batch']):
        tensor = torch.from_numpy(data['events'][start:start + p['batch']])
        batch = [row['nodes'].cpu().numpy() for row in final_query_candidates(model, tensor, p['max_rounds'], p['prior_rounds'])]
        values.append(np.stack(batch, axis=1))
    return np.concatenate(values)


@torch.no_grad()
def lazy_replay(model, data, trace, p):
    rows = []
    for i in range(p['lazy_replay_count']):
        ev = records(data['events'][i])
        query = ev[-1]['child']
        tensor = torch.from_numpy(data['events'][i:i + 1])
        for method, verifier in [('component', old_verify), ('path', verify_path)]:
            t0 = time.perf_counter()
            steps, selected = 0, None
            for row in final_query_candidates(model, tensor, p['max_rounds'], p['prior_rounds']):
                steps += 1
                nodes = row['nodes'][0].cpu().numpy()
                # Batch-independent forward can differ by float roundoff only.
                np.testing.assert_allclose(nodes, trace[i, steps - 1], rtol=2e-5, atol=2e-6)
                selected = verifier({str(j): float(v) for j, v in enumerate(nodes)}, ev, query, p['primary_threshold'])
                if selected['status'] == 'VERIFIED':
                    break
            rows.append(dict(example_id=i, method=method, actual_refinements=steps,
                             prior_refinements=2 * p['prior_rounds'], actual_seconds=time.perf_counter() - t0,
                             accepted=selected['status'] == 'VERIFIED', bound=selected['bound']))
    return rows


def evaluate(out):
    out = Path(out)
    p = verify_frozen(out)
    if (out / 'completion.json').exists():
        return verify_completion(out, p)
    torch.set_num_threads(1)
    write(out / 'challenges.json', challenge_checks())
    receipts = []
    for split, data_seed in p['sources'].items():
        data = make_data(p['count'], data_seed, split)
        data_path = out / f'data-{split}.npz'
        if not data_path.exists():
            np.savez_compressed(data_path, **data)
            write(out / f'source-open-{split}.json', dict(opened_utc=datetime.now(timezone.utc).isoformat(),
                  source_seed=data_seed, source_sha256=sha(data_path), protocol_sha256=sha(out / 'protocol.json')))
        else:
            saved = np.load(data_path)
            for key in data:
                np.testing.assert_array_equal(data[key], saved[key])
        events = [records(tape) for tape in data['events']]
        queries = [ev[-1]['child'] for ev in events]
        targets = [reference_values(ev)[q] for ev, q in zip(events, queries)]
        for seed in p['model_seeds']:
            receipt_path = out / f'receipt-{split}-{seed}.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                verify_receipt(out, p, receipt, split, seed)
                receipts.append(receipt)
                continue
            started = time.perf_counter()
            model = load_model(OLD / 'learning/checkpoints' / p['checkpoint'].format(seed=seed))
            t0 = time.perf_counter()
            nodes = traces(model, data, p)
            inference_seconds = time.perf_counter() - t0
            bounds = np.full((p['count'], p['max_rounds'], 2), np.nan, np.float64)
            correct = np.zeros((p['count'], p['max_rounds']), bool)
            acceptance = np.zeros((p['count'], p['max_rounds'], 2, len(p['thresholds'])), bool)
            audit_count = 0
            certs = []
            for i, (ev, query, target) in enumerate(zip(events, queries, targets)):
                for k in range(p['max_rounds']):
                    candidate = {str(j): float(v) for j, v in enumerate(nodes[i, k])}
                    correct[i, k] = abs(Fraction(candidate[query]) - target) <= Fraction(1, 10)
                    old = old_verify(candidate, ev, query, p['primary_threshold'])
                    new = verify_envelope(bind_candidate(candidate, ev, query, f'{split}:{i}'), ev, query, f'{split}:{i}', p['primary_threshold'])
                    for j, result in enumerate([old, new]):
                        if result['bound'] is not None:
                            bounds[i, k, j] = result['bound']
                        for threshold_index, acceptance_threshold in enumerate(p['thresholds']):
                            if j == 0:
                                accept = result['bound'] is not None and result['bound'] <= acceptance_threshold
                            else:
                                exact = result.get('bound_exact')
                                accept = exact is not None and Fraction(**exact) <= Fraction(str(acceptance_threshold))
                            acceptance[i, k, j, threshold_index] = accept
                    if new['status'] == 'VERIFIED':
                        if not correct[i, k] or not audit_certificate(new['certificate'], candidate, ev, query, p['primary_threshold']):
                            raise AssertionError('SOUNDNESS_OR_CERTIFICATE_FAILURE')
                        audit_count += 1
                        certs.append(dict(example_id=i, round=k + 1, certificate=new['certificate']))
            raw_path = out / f'raw-{split}-{seed}.npz'
            np.savez_compressed(raw_path, nodes=nodes, bounds=bounds, correct=correct, acceptance=acceptance,
                                targets=np.array([float(x) for x in targets]), query_ids=np.array([int(x) for x in queries]),
                                **{k: data[k] for k in ['example_id', 'affected', 'inverse', 'depth']})
            cert_path = out / f'certificates-{split}-{seed}.json'
            write(cert_path, certs)
            replay_path = out / f'lazy-{split}-{seed}.json'
            write(replay_path, lazy_replay(model, data, nodes, p))
            receipt = dict(split=split, seed=seed, source_sha256=sha(data_path),
                           protocol_sha256=sha(out / 'protocol.json'), audited_certificates=audit_count,
                           full_trace_inference_seconds=inference_seconds, total_seconds=time.perf_counter() - started,
                           artifacts={path.name: sha(path) for path in [raw_path, cert_path, replay_path]})
            write(receipt_path, receipt)
            receipts.append(receipt)
            print(json.dumps(dict(completed=f'{split}-{seed}', seconds=receipt['total_seconds'], certified=audit_count)), flush=True)
    result = dict(status='MEASURED', protocol_sha256=sha(out / 'protocol.json'), receipts=receipts,
                  independent_source_problems=p['count'] * len(p['sources']),
                  model_problem_executions=p['count'] * len(p['sources']) * len(p['model_seeds']))
    write(out / 'completion.json', result)
    return result


def summarize(out):
    out = Path(out)
    p = verify_frozen(out)
    completion = verify_completion(out, p)
    rows, paired = [], []
    for receipt in completion['receipts']:
        for rel, expected in receipt['artifacts'].items():
            if sha(out / rel) != expected:
                raise RuntimeError('RESULT_HASH_MISMATCH')
        raw = np.load(out / f"raw-{receipt['split']}-{receipt['seed']}.npz")
        bounds, correct = raw['bounds'], raw['correct']
        for threshold_index, threshold_value in enumerate(p['thresholds']):
            for j, method in enumerate(['component', 'path']):
                accepted_all = raw['acceptance'][:, :, j, threshold_index]
                for policy in ['fixed8', 'adaptive']:
                    chosen = np.full(len(bounds), p['max_rounds'] - 1, int)
                    if policy == 'adaptive':
                        has = accepted_all.any(axis=1)
                        chosen[has] = accepted_all[has].argmax(axis=1)
                    selected_accept = accepted_all[np.arange(len(bounds)), chosen]
                    selected_correct = correct[np.arange(len(bounds)), chosen]
                    rows.append(dict(split=receipt['split'], seed=receipt['seed'], method=method, policy=policy,
                                     threshold=threshold_value, mean_final_refinements=float((chosen + 1).mean()),
                                     **metrics(selected_accept, selected_correct)))
        rng = np.random.default_rng(p['bootstrap_seed'])
        main_index = p['thresholds'].index(p['primary_threshold'])
        delta = raw['acceptance'][:, -1, 1, main_index].astype(int) - raw['acceptance'][:, -1, 0, main_index].astype(int)
        bootstrap = delta[rng.integers(0, len(delta), (p['bootstrap_repetitions'], len(delta)))].mean(axis=1)
        paired.append(dict(split=receipt['split'], seed=receipt['seed'],
                           path_minus_component_coverage=float(delta.mean()),
                           source_paired_bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist(),
                           gained=int((delta > 0).sum()), lost=int((delta < 0).sum())))
    result = dict(status='MEASURED', rows=rows, paired_primary=paired,
                  inference_note='No seed pooling: each source problem is shared across model seeds. Trace replay estimates halt rounds; lazy subset measures actual calls.',
                  risk_note='Zero accepted errors is finite evidence; null risk denotes zero accepted. Exact proof remains conditional on source/affine contract.')
    write(out / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze(args.out)
    if args.evaluate:
        evaluate(args.out)
    if args.summarize:
        summarize(args.out)


if __name__ == '__main__':
    main()
