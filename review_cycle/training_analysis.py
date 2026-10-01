"""Prespecified R2-R5 contrasts from sealed raw predictions, never training loss."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from . import training as t


def seed_summary(differences):
    if len(differences) != 3 or not all(math.isfinite(x) for x in differences):
        raise ValueError('exactly three finite paired seed effects required')
    mean = statistics.mean(differences)
    half = 4.302652729 * statistics.stdev(differences) / math.sqrt(3)
    return dict(seed_effects=differences, mean=mean, approximate_seed_t95=[mean - half, mean + half],
                seeds=3, interval_scope='three-seed t approximation; not multiplicity adjusted',
                direction='POSITIVE_INTERVAL' if mean - half > 0 else 'NEGATIVE_INTERVAL' if mean + half < 0 else 'INCLUDES_ZERO')


def paired_accuracy(candidate, baseline):
    for key in ('inputs', 'targets', 'input_ids'):
        if not np.array_equal(candidate[key], baseline[key]):
            raise ValueError('UNPAIRED_RAW_' + key)
    a = candidate['predictions'][:, -1] == candidate['targets'][:, -1]
    b = baseline['predictions'][:, -1] == baseline['targets'][:, -1]
    return dict(n=len(a), candidate_correct=int(a.sum()), baseline_correct=int(b.sum()),
                gained=int((a & ~b).sum()), lost=int((~a & b).sum()),
                difference=float(a.mean() - b.mean()))


class Evidence:
    def __init__(self, out):
        self.out = Path(out)
        self.manifest = t.verify_frozen(out)
        self.p = self.manifest['protocol']
        selection = t._read_selection(out, self.manifest)
        self.conditions = t.conditions(self.p) + t.large_conditions(self.p, selection['selected_lr'], confirm=True)
        all_training = self.conditions + t.large_conditions(self.p)
        if len(all_training) != 48:
            raise ValueError('INCOMPLETE_DECLARED_TRAINING_SET')
        for condition in all_training:
            t._read_complete(out, condition, self.manifest)
        evaluation = json.loads((self.out / 'evaluation.json').read_text())
        if evaluation['status'] != 'COMPLETE' or evaluation['protocol_hash'] != self.manifest['protocol_hash']:
            raise ValueError('FULL_FINAL_EVALUATION_REQUIRED')
        expected_variants = set()
        for condition in self.conditions:
            cid = t.ident(condition)
            for variant in (['selected', 'equal-label'] if condition['supervision'] == 'dense' else ['selected']):
                key = cid if variant == 'selected' else cid + '-equal-label'
                expected_variants.add(key)
                metadata = json.loads((self.out / cid / (variant + '-evaluation.json')).read_text())
                checkpoint = self.out / cid / (variant + '.pt')
                t._validate_evaluation(metadata, condition, t.execution_protocol(self.p, condition), self.manifest,
                                       self.out, variant, checkpoint)
                if evaluation['conditions'].get(key) != metadata:
                    raise ValueError('FINAL_AGGREGATE_DIFFERS_FROM_SOURCE')
        if set(evaluation['conditions']) != expected_variants:
            raise ValueError('FINAL_VARIANT_SET_MISMATCH')
        self.selection = selection
        self.raws = {}

    def condition(self, task, seed, mode='rlt', supervision='dense', arrangement='random', stage='small'):
        found = [c for c in self.conditions if all(c[k] == v for k, v in
                dict(task=task, seed=seed, mode=mode, supervision=supervision, arrangement=arrangement, stage=stage).items())]
        if len(found) != 1:
            raise ValueError('CONDITION_IDENTITY_NOT_UNIQUE')
        return found[0]

    def raw(self, condition, endpoint, variant='selected'):
        key = (t.ident(condition), endpoint, variant)
        if key in self.raws:
            return self.raws[key]
        metadata = json.loads((self.out / key[0] / (variant + '-evaluation.json')).read_text())
        if metadata['condition'] != condition or metadata['variant'] != variant:
            raise ValueError('EVALUATION_IDENTITY_MISMATCH')
        if metadata['protocol_hash'] != self.manifest['protocol_hash']:
            raise ValueError('EVALUATION_PROTOCOL_MISMATCH')
        if endpoint.startswith('L'):
            matched = [r for r in metadata['lengths'] if r['length'] == int(endpoint[1:])]
        else:
            _, number, placement = endpoint.split('-')
            matched = [r for r in metadata['neutral'] if r['slots'] == int(number) and r['placement'] == placement]
        if len(matched) != 1:
            raise ValueError('ENDPOINT_INVENTORY_MISMATCH')
        row = matched[0]
        path = self.out / row['raw_file']
        if t.sha(path) != row['sha256']:
            raise ValueError('RAW_HASH_MISMATCH')
        with np.load(path) as loaded:
            data = {k: loaded[k] for k in loaded.files}
        self.raws[key] = data
        return data

    def contrast(self, rid, name, task, endpoint, candidate_spec, baseline_spec,
                 candidate_variant='selected', baseline_variant='selected', primary=False, interpretation=None):
        rows = []
        for seed in self.p['primary_seeds']:
            ca = self.condition(task, seed, **candidate_spec)
            cb = self.condition(task, seed, **baseline_spec)
            comparison = paired_accuracy(self.raw(ca, endpoint, candidate_variant), self.raw(cb, endpoint, baseline_variant))
            rows.append(dict(seed=seed, candidate=t.ident(ca), baseline=t.ident(cb), **comparison))
        return dict(research_id=rid, contrast=name, task=task, endpoint=endpoint, primary=primary,
                    candidate_variant=candidate_variant, baseline_variant=baseline_variant,
                    interpretation=interpretation, per_seed=rows, **seed_summary([r['difference'] for r in rows]))


def analyze(out):
    evidence = Evidence(out)
    p = evidence.p
    contrasts = []
    endpoints = [f'L{x}' for x in p['test_lengths']] + ['neutral-0-base'] + [
        f'neutral-{slots}-{placement}' for slots in p['neutral_slots'] if slots
        for placement in ('prefix', 'middle', 'suffix')]
    for task in ('parity', 'five'):
        for endpoint in endpoints:
            contrasts.append(evidence.contrast('R2', 'blocked-minus-random', task, endpoint,
                dict(arrangement='blocked'), {}, primary=endpoint == 'neutral-128-suffix',
                interpretation='same training token histograms/nonneutral order; long neutral run remains distribution shift'))
    for mode in ('rlt', 'nofeedback'):
        for length in p['test_lengths']:
            contrasts.append(evidence.contrast('R3', f'{mode}-final-minus-sparse', 'parity', f'L{length}',
                dict(mode=mode, supervision='final'), dict(mode=mode, supervision='sparse'), primary=length == 128,
                interpretation='same updates, inputs and one-label budget; only supervised position policy differs'))
        for supervision in ('sparse', 'final'):
            for length in p['test_lengths']:
                contrasts.append(evidence.contrast('R3', f'{mode}-{supervision}-minus-dense', 'parity', f'L{length}',
                    dict(mode=mode, supervision=supervision), dict(mode=mode), primary=length == 128,
                    interpretation='same updates/inputs; sparse or final has one label, dense32'))
                contrasts.append(evidence.contrast('R3', f'{mode}-{supervision}2048-minus-dense64', 'parity', f'L{length}',
                    dict(mode=mode, supervision=supervision), dict(mode=mode), baseline_variant='equal-label',
                    primary=False, interpretation='equal65536labels; different inputs/updates/compute; not isolated supervision-density effect'))
    for supervision in ('dense', 'sparse', 'final'):
        for length in p['test_lengths']:
            contrasts.append(evidence.contrast('R3', f'rlt-minus-nofeedback-{supervision}', 'parity', f'L{length}',
                dict(supervision=supervision), dict(mode='nofeedback', supervision=supervision), primary=length == 128,
                interpretation='same input, target budget and initial allocated weights; active parameter routes differ'))
    for supervision in ('dense', 'final'):
        for length in p['test_lengths']:
            contrasts.append(evidence.contrast('R4', f'full-minus-tbptt8-{supervision}', 'five', f'L{length}',
                dict(supervision=supervision), dict(mode='tbptt8', supervision=supervision), primary=length == 128,
                interpretation='only s/dk/dv truncated; global encoder/mk/mv gradients remain'))
    confirms = []
    for seed in p['confirm_seeds']:
        c = evidence.condition('parity', seed, stage='confirm')
        receipt = t._read_complete(out, c, evidence.manifest)
        rows = []
        for endpoint in endpoints:
            raw = evidence.raw(c, endpoint)
            prediction, target = raw['predictions'][:, -1], raw['targets'][:, -1]
            counts = np.bincount(prediction, minlength=2)
            rows.append(dict(endpoint=endpoint, n=len(target), correct=int((prediction == target).sum()),
                             accuracy=float((prediction == target).mean()), prediction_histogram=counts.tolist(),
                             target_histogram=np.bincount(target, minlength=2).tolist(),
                             observed_constant_prediction=bool(np.count_nonzero(counts) == 1)))
        confirms.append(dict(seed=seed, condition=t.ident(c), selected_step=receipt['selected_step'],
                             selected_lr=c['lr'], endpoints=rows))
    result = dict(protocol_hash=evidence.manifest['protocol_hash'], analysis_plan_sha256=t.sha(t.ROOT / 'docs/training-analysis-plan.md'),
                  status='MEASURED', complete_training_conditions=48, evaluated_models=39,
                  selected_lr=evidence.selection['selected_lr'], contrasts=contrasts, R5_confirmation=confirms,
                  caveat='Predeclared multiple endpoints; exploratory uncertainty intervals, no global superiority claim. Missing conditions fail rather than disappearing from denominators.')
    t.write_json(Path(out) / 'scientific-analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = analyze(args.out)
    print(json.dumps(dict(status=result['status'], contrasts=len(result['contrasts']), confirmatory=len(result['R5_confirmation']))))


if __name__ == '__main__':
    main()
