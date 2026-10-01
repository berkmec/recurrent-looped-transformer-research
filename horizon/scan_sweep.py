"""U5-scan: run the exact-preservation gate twice, changing only what the gate is allowed to read.

The question comes from putting our own measurement beside a line that already solved the same shape of problem.
Ours: 2068 bytes/s at 30M, because the decoder loop is sequential over 512 timesteps. Theirs (Mamba, S4): the
same limit, defeated by a parallel scan - which needs `s_t = a_t*s_{t-1} + b_t` with both coefficients free of s.
Our gate reads `[e_t, s_{t-1}]`, so it is not. Mamba's selectivity reads the input alone, so it is.

Hence the cycle: does the campaign's one replicated positive survive the gate losing its view of the state?

  gate_state  the sealed U2c hard_st gate, reproduced here bit for bit
  gate_input  the same thing with the head reading e_t only

BOTH ARMS RUN IN THIS CYCLE rather than comparing against U2c's stored numbers. Same machine, same seeds, same
protocol, one changed term - the method rule this campaign learned in U4-sep2 and then failed to apply in
U4-inject, where the arms turned out to differ in two terms and the headline had to be bounded afterwards.

What a positive would and would not license, stated before any data: an input-only gate is NECESSARY for a
parallel scan and NOT SUFFICIENT, because the candidate still carries the Core's additive feedback from
s_{t-1}. A surviving result makes the second and larger change worth testing; it does not make the model
scannable.

`horizon/gated_sweep.py` and `core/model.py` are hashed inside sealed manifests. Everything reusable is imported
from them; only the model factory and the arm names are new, and `tests/test_scanable.py` proves the variant is
bit-identical to the sealed gate in the arm that stands for it.

Usage: python -B -m horizon.scan_sweep --freeze --out results/U5scan
       python -B -m horizon.scan_sweep --run-all --out results/U5scan
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from horizon import gated as gd
from horizon import gated_sweep as gs
from horizon import guards as gu
from horizon import scanable as sc
from horizon import sweep as sw
from horizon import task_families as tf

ARMS = ('gate_state', 'gate_input')
SOURCES = gs.SOURCES + ('horizon/gated.py', 'horizon/scanable.py', 'horizon/scan_sweep.py')


def protocol():
    """U2c's protocol with a different arm list. Nothing else moves, so the two cycles stay comparable."""
    p = dict(gs.protocol())
    p.update(version=1, arms=list(ARMS), gate_loss_weight=0.,
             purpose='U5-scan: is the gate dependence on the committed state load-bearing, or can the gate read '
                     'the input alone? An input-only gate is the cheaper of the two conditions a parallel scan '
                     'needs; the candidate feedback is the other and is not touched here.',
             preregistration=dict(
                 prediction='exact preservation survives the gate losing its view of the state, because the '
                            'campaign measured the VALUE of exact preservation, not of state-conditioned '
                            'gating - U2c never varied what the head reads',
                 falsification='if gate_input repair falls more than 15 points below gate_state on either task '
                               'in 2 of 3 seeds, the state dependence is load-bearing and the scan route closes',
                 thresholds=dict(survives_pp=15., seeds_required=2, tasks_required=2),
                 licenses='a positive makes removing the candidate feedback worth testing; it does NOT make the '
                          'model scannable, since that second state dependence remains'))
    return p


def mode_of(arm):
    return 'hard_st'                      # both arms are the replicated positive; only the gate input differs


def gate_input_of(arm):
    return 'input' if arm == 'gate_input' else 'state'


def conditions(p):
    """Arms x tasks x seeds, and optionally x learning rates.

    `lrs` is absent from the base protocol, so U5scan's twelve conditions and their idents are untouched. When it
    is present, a condition at a rate other than the protocol's own carries a suffix - which keeps the directory
    names of an existing cycle stable while a new one varies the axis that decided the solution family three times
    in this campaign's history.
    """
    rates = list(p.get('lrs') or [p['lr']])
    return [dict(arm=arm, task=task, seed=seed, width=p['width'], heads=p['heads'], window=p['window'],
                 lr=lr, stage='u5scan', lr_tag=(None if lr == p['lr'] else 'lr%g' % lr))
            for arm in p['arms'] for task in p['tasks'] for seed in p['primary_seeds'] for lr in rates]


def ident(c):
    base = 'u5-%s-%s-w%d-s%d' % (c['arm'], c['task'], c['width'], c['seed'])
    return base if not c.get('lr_tag') else base + '-' + c['lr_tag']


def make_model(c, p):
    core = sw.make_model(c, p)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'] + 7)          # the same stream U2c used, so the cores start identically
        return sc.ScanGatedCore(core, mode=mode_of(c['arm']), k=p['gate_k'], gate_input=gate_input_of(c['arm']))


def source_hashes():
    return {name: sw.sha(gs.ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = dict(p or protocol())
    manifest = dict(protocol=p, protocol_hash=sw.digest(p), sources=source_hashes(), frozen_utc=time.time())
    if (out / 'frozen.json').exists():
        old = verify_frozen(out)
        if old['protocol_hash'] != manifest['protocol_hash']:
            raise ValueError('refusing frozen protocol change')
        return old
    if any(out.iterdir()):
        raise ValueError('new freeze requires empty output directory')
    sw.write_json(out / 'frozen.json', manifest)
    (out / 'frozen.sha256').write_text(sw.sha(out / 'frozen.json'), encoding='ascii')
    return manifest


def verify_frozen(out):
    out = Path(out)
    if sw.sha(out / 'frozen.json') != (out / 'frozen.sha256').read_text(encoding='ascii').strip():
        raise ValueError('frozen manifest hash mismatch')
    manifest = json.loads((out / 'frozen.json').read_text(encoding='utf8'))
    if manifest['protocol_hash'] != sw.digest(manifest['protocol']) or manifest['sources'] != source_hashes():
        raise ValueError('protocol/source drift')
    torch.set_num_threads(manifest['protocol'].get('threads', 1))
    return manifest


def verify_evidence(out):
    """For ANALYSIS only: the protocol and every condition's own hashes, with source drift recorded not hidden.

    verify_frozen stays in force for training, so no condition can be trained against changed code. Here the runs
    are finished and their bytes are hashed inside their own receipts; refusing to read them because the analysis
    code was added afterwards would protect nothing. Refusing to read them because the registered THRESHOLDS
    moved would protect everything, so that check is kept.
    """
    out = Path(out)
    if sw.sha(out / 'frozen.json') != (out / 'frozen.sha256').read_text(encoding='ascii').strip():
        raise ValueError('frozen manifest hash mismatch')
    manifest = json.loads((out / 'frozen.json').read_text(encoding='utf8'))
    if manifest['protocol_hash'] != sw.digest(manifest['protocol']):
        raise ValueError('the registered protocol no longer matches its hash; nothing here may be read')
    now = source_hashes()
    drift = sorted(n for n, h in manifest['sources'].items() if now.get(n) != h)
    problems = []
    for c in conditions(manifest['protocol']):
        d = out / ident(c)
        receipt = d / 'train.json'
        if not receipt.exists():
            problems.append(ident(c) + ': no train.json'); continue
        t = json.loads(receipt.read_text(encoding='utf8'))
        if t['protocol_hash'] != manifest['protocol_hash']:
            problems.append(ident(c) + ': trained under a different protocol')
        for name, key in (('selected.pt', 'selected_sha256'), ('updates.jsonl', 'updates_sha256')):
            if sw.sha(d / name) != t[key]:
                problems.append(f'{ident(c)}: {name} does not match its receipt')
    if problems:
        raise ValueError('evidence integrity failed:\n  ' + '\n  '.join(problems))
    torch.set_num_threads(manifest['protocol'].get('threads', 1))
    return manifest, drift


def run_one(out, c):
    """The U2c training loop, with this cycle's model factory. Structure kept identical on purpose."""
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    if ident(c) not in {ident(x) for x in conditions(p)}:
        raise ValueError('unknown condition ' + ident(c))
    directory = out / ident(c); directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        return json.loads((directory / 'train.json').read_text(encoding='utf8'))
    if any(directory.iterdir()):
        raise ValueError('incomplete condition directory requires audit; no automatic reuse: ' + str(directory))
    (directory / 'active.lock').write_text(json.dumps(dict(pid=os.getpid(), time=time.time())), encoding='utf8')
    try:
        wall = time.perf_counter()
        model = make_model(c, p)
        opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=p['weight_decay'])
        torch.save(sw.cpu_state(model), directory / 'initial.pt')
        vx, vy, _ = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'validation', p['validation_n'], p['train_length'])
        counter = gu.GuardCounter(); best, selected, train_seconds = None, None, 0.
        for step in range(1, p['steps'] + 1):
            start = time.perf_counter()
            batch = tf.train_batch(p['source_seeds'], c['task'], c['seed'], step, p['batch'], p['train_length'])
            model.train()
            logits, gates, glogits = model(batch['x'])
            loss = F.cross_entropy(logits.flatten(0, 1), batch['y'].flatten())   # no gate supervision in either arm
            report = counter.record(gu.guarded_step(opt, model.parameters(), loss, clip_norm=p['clip']))
            train_seconds += time.perf_counter() - start
            row = dict(step=step, applied=report['applied'], loss=report['loss'],
                       task_loss=float(loss.detach()) if math.isfinite(float(loss.detach())) else None,
                       grad_norm=report['grad_norm_before_clip'],
                       hold_rate=float((gates.detach() <= .5).float().mean()),
                       input_hash=batch['input_hash'], source_seed=batch['source_seed'])
            if step % p['validation_every'] == 0 or step == p['steps']:
                stats, _, _ = gs.predict(model, vx, vy, p); row['validation'] = gs.json_safe(stats)
                if best is None or stats['final_loss'] < best:
                    best = stats['final_loss']; selected = dict(step=step, model=sw.cpu_state(model), validation=stats)
            sw.append_line(directory / 'updates.jsonl', gs.json_safe(row))
        torch.save(selected['model'], directory / 'selected.pt'); torch.save(sw.cpu_state(model), directory / 'final.pt')
        result = dict(condition=c, ident=ident(c), steps=p['steps'], selected_step=selected['step'],
                      selected_validation=gs.json_safe(selected['validation']), guard=counter.summary(),
                      train_seconds=train_seconds, wall_seconds=time.perf_counter() - wall,
                      protocol_hash=manifest['protocol_hash'], selected_sha256=sw.sha(directory / 'selected.pt'),
                      initial_sha256=sw.sha(directory / 'initial.pt'), updates_sha256=sw.sha(directory / 'updates.jsonl'))
        sw.write_json(directory / 'train.json', gs.json_safe(result))
        sw.write_json(directory / 'complete.json', dict(ident=ident(c), utc=time.time(), train_sha256=sw.sha(directory / 'train.json')))
        return result
    finally:
        (directory / 'active.lock').unlink(missing_ok=True)


def evaluate_one(out, c):
    """The U2c evaluation on this cycle's conditions: neutral runs at every slot and placement, plus clean."""
    from horizon import metrics as mt
    out = Path(out); manifest, _ = verify_evidence(out); p = manifest['protocol']
    directory = out / ident(c); path = directory / 'evaluation.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    if not (directory / 'complete.json').exists():
        raise ValueError('evaluation requires a complete condition: ' + ident(c))
    model = make_model(c, p); model.load_state_dict(torch.load(directory / 'selected.pt', weights_only=True))
    t0 = time.perf_counter()
    row = dict(condition=c, ident=ident(c), protocol_hash=manifest['protocol_hash'],
               selected_sha256=sw.sha(directory / 'selected.pt'), neutral={}, clean={})
    x, y, seed = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'neutral', p['n_neutral'], p['train_length'])
    stats, base_pred, gates = gs.predict(model, x, y, p)
    row['neutral']['0-base'] = dict(final_accuracy=stats['final_accuracy'],
                                    gate=gd.gate_rates(gates, gd.gate_target(x, p['gate_k'])), source_seed=seed)
    for slots in p['neutral_slots']:
        for placement in p['placements']:
            xx, yy = tf.insert_neutrals(x, c['task'], slots, placement)
            stats, pred, gates = gs.predict(model, xx, yy, p)
            row['neutral'][f'{slots}-{placement}'] = dict(
                final_accuracy=stats['final_accuracy'],
                gate=gd.gate_rates(gates, gd.gate_target(xx, p['gate_k'])),
                **mt.flip_counts(base_pred[:, -1].numpy(), pred[:, -1].numpy(), yy[:, -1].numpy()))
    for L in p['clean_lengths']:
        cx, cy, cseed = tf.heldout(p['source_seeds'], c['task'], c['seed'], 'test', p['n_clean'], L)
        stats, pred, gates = gs.predict(model, cx, cy, p)
        row['clean'][str(L)] = dict(final_accuracy=stats['final_accuracy'], prefix_accuracy=stats['all_accuracy'],
                                    gate=gd.gate_rates(gates, gd.gate_target(cx, p['gate_k'])), source_seed=cseed)
    row['seconds'] = time.perf_counter() - t0
    sw.write_json(path, gs.json_safe(row))
    return row


ENDPOINT = '512-suffix'


def repair_fraction(accuracy, none, ceiling):
    """The scale every earlier cycle used: 0 % is the untreated model, 100 % is the run-aware freeze ceiling."""
    span = ceiling - none
    return None if abs(span) < 1e-9 else (accuracy - none) / span * 100.


def collect(out):
    out = Path(out); manifest, drift = verify_evidence(out); p = manifest['protocol']
    reference = gs.reference_rows(p)
    ceiling = {}
    for path in Path(p['reference']).glob('u1-*.json'):
        r = json.loads(path.read_text(encoding='utf8'))
        arms = r['arms']
        pick = next((k for k in ('freeze_k2', 'freeze-k2', 'freeze_run_k2') if k in arms), None)
        if pick:
            ceiling[(r['task'], r['seed'])] = arms[pick]
    rows = {}
    for c in conditions(p):
        e = evaluate_one(out, c)
        t = json.loads((out / ident(c) / 'train.json').read_text(encoding='utf8'))
        key = (c['task'], c['seed'])
        none = reference.get(key, {}).get('neutral', {}).get(ENDPOINT, {}).get('final_accuracy')
        top = ceiling.get(key, {}).get('neutral', {}).get(ENDPOINT, {}).get('final_accuracy')
        rows[ident(c)] = dict(
            arm=c['arm'], task=c['task'], seed=c['seed'], lr=c['lr'],
            accuracy=e['neutral'][ENDPOINT]['final_accuracy'],
            validation=t['selected_validation']['final_accuracy'],
            clean4096=e['clean']['4096']['final_accuracy'],
            hold_rate=e['neutral'][ENDPOINT]['gate'].get('hold_rate'),
            none=none, ceiling=top,
            repair=repair_fraction(e['neutral'][ENDPOINT]['final_accuracy'], none, top)
            if (none is not None and top is not None) else None)
    return manifest, rows, drift


def verdict(rows, thresholds):
    """The rule as registered: gate_input must stay within `survives_pp` of gate_state on enough seeds and tasks.

    Learning is checked first. A collapsed arm cannot be read as a geometry result - the same reasoning that gave
    the two preceding cycles their INTERVENTION_INEFFECTIVE and NOT_MEASURABLE branches. A seed where the
    INPUT-ONLY arm failed to learn is a failure OF that arm and counts against it; a seed where the STATE arm
    failed would make the pair unreadable and is reported separately.
    """
    pp, need_seeds, need_tasks = thresholds['survives_pp'], thresholds['seeds_required'], thresholds['tasks_required']
    by_task, detail, unreadable = {}, [], []
    tasks = sorted({r['task'] for r in rows.values()})
    for task in tasks:
        held = 0
        for seed in sorted({r['seed'] for r in rows.values() if r['task'] == task}):
            a = next((r for r in rows.values() if r['task'] == task and r['seed'] == seed and r['arm'] == 'gate_state'), None)
            b = next((r for r in rows.values() if r['task'] == task and r['seed'] == seed and r['arm'] == 'gate_input'), None)
            if not a or not b or a['repair'] is None or b['repair'] is None:
                unreadable.append(f'{task}/s{seed}: missing repair'); continue
            gap = a['repair'] - b['repair']
            ok = gap <= pp
            held += int(ok)
            detail.append(dict(task=task, seed=seed, state_repair=a['repair'], input_repair=b['repair'],
                               gap_pp=gap, within=ok, state_val=a['validation'], input_val=b['validation'],
                               input_learned=b['validation'] > .5, state_learned=a['validation'] > .5))
        by_task[task] = held
    survived = sum(1 for t in tasks if by_task[t] >= need_seeds)
    collapsed_input = [d for d in detail if not d['input_learned']]
    collapsed_state = [d for d in detail if not d['state_learned']]
    if collapsed_state:
        status = 'PAIR_UNREADABLE'
    elif survived >= need_tasks and not collapsed_input:
        status = 'STATE_DEPENDENCE_NOT_LOAD_BEARING'
    elif survived >= need_tasks and collapsed_input:
        status = 'SURVIVES_BUT_LESS_STABLE'
    else:
        status = 'STATE_DEPENDENCE_LOAD_BEARING'
    return dict(status=status, thresholds=thresholds, seeds_within_by_task=by_task, tasks_survived=survived,
                collapsed_input_arm=[f"{d['task']}/s{d['seed']}" for d in collapsed_input],
                collapsed_state_arm=[f"{d['task']}/s{d['seed']}" for d in collapsed_state],
                unreadable=unreadable, pairs=detail,
                meaning=dict(
                    STATE_DEPENDENCE_NOT_LOAD_BEARING='the gate can read the input alone without losing repair: '
                        'the first of two conditions a parallel scan needs is satisfied, and removing the '
                        "candidate's feedback becomes the next question. The model is NOT yet scannable.",
                    SURVIVES_BUT_LESS_STABLE='repair survives where the arm learns, but the input-only gate failed '
                        'to learn on at least one seed. Cheaper recurrence at the cost of training stability, and '
                        'the instability has to be characterised before it can be called a win.',
                    STATE_DEPENDENCE_LOAD_BEARING='removing the gate view of the state costs more than the '
                        'registered margin: the sequential cost is structural and the scan route closes here.',
                    PAIR_UNREADABLE='the state arm itself failed to learn on some seed, so those pairs compare '
                        'nothing; fix the control before reading the treatment.')[status])


def lr_report(out):
    """For the lr axis: did each arm learn at each rate? The question is stability, not repair."""
    manifest, rows, drift = collect(out)
    table = sorted(rows.values(), key=lambda r: (r['task'], r['seed'], r['arm'], r['lr']))
    learned = {}
    for r in table:
        learned[(r['arm'], r['lr'])] = learned.get((r['arm'], r['lr']), 0) + int(r['validation'] > .5)
    report = dict(kind='U5SCAN_LR', utc=time.time(), protocol_hash=manifest['protocol_hash'],
                  rows=rows, learned_by_arm_and_lr={'%s@%g' % k: v for k, v in sorted(learned.items())},
                  source_drift=drift,
                  question='is the input-only gate collapse at five/s143 a learning-rate phenomenon? A rate at '
                           'which the input-only arm trains means the gate is not broken, it is rate-sensitive, '
                           'which is the pattern this campaign has already measured three times.')
    sw.write_json(Path(out, 'lr-report.json'), gs.json_safe(report))
    return report


def analyze(out):
    manifest, rows, drift = collect(out)
    v = verdict(rows, manifest['protocol']['preregistration']['thresholds'])
    report = dict(kind='U5SCAN_ANALYSIS', utc=time.time(), protocol_hash=manifest['protocol_hash'],
                  endpoint=ENDPOINT, rows=rows, verdict=v,
                  preregistration=manifest['protocol']['preregistration'],
                  source_drift=drift,
                  source_drift_note='files hashed at freeze time whose bytes changed afterwards. The evaluation '
                                    'and the verdict rule were appended to horizon/scan_sweep.py AFTER the 12 '
                                    'conditions finished training, so this drift is expected and is recorded '
                                    'rather than hidden. The registered thresholds and predictions are inside '
                                    'protocol_hash and did NOT change; training still refuses to run against '
                                    'drifted sources.')
    sw.write_json(Path(out, 'analysis.json'), gs.json_safe(report))
    return report


def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--run-all', action='store_true')
    group.add_argument('--analyze', action='store_true')
    group.add_argument('--lr-report', action='store_true', dest='lr_report')
    parser.add_argument('--out', default='results/U5scan')
    parser.add_argument('--only', default=None)
    args = parser.parse_args()
    if args.freeze:
        m = freeze(args.out)
        print(json.dumps(dict(out=args.out, protocol_hash=m['protocol_hash'], arms=m['protocol']['arms'],
                              conditions=len(conditions(m['protocol']))))); return
    if args.analyze:
        r = analyze(args.out)
        v = r['verdict']
        print('%-26s %-11s %9s %9s %8s %7s' % ('cift', 'gorev/tohum', 'state', 'input', 'fark', 'icinde'))
        for d in v['pairs']:
            print('%-26s %-11s %9.1f %9.1f %+8.1f %7s' % (f"{d['task']}/s{d['seed']}", '', d['state_repair'],
                                                          d['input_repair'], -d['gap_pp'], d['within']))
        print()
        if r['source_drift']:
            print('source drift (expected, recorded):', ', '.join(r['source_drift']))
        print('VERDICT', v['status'])
        print(' ', v['meaning'])
        if v['collapsed_input_arm']:
            print('  input-only arm failed to learn on:', ', '.join(v['collapsed_input_arm']))
        print('written', Path(args.out, 'analysis.json'))
        return
    if args.lr_report:
        r = lr_report(args.out)
        print('%-30s %6s %10s %10s %9s' % ('kosul', 'lr', 'dogrulama', 'onarim', 'hold'))
        for k in sorted(r['rows']):
            x = r['rows'][k]
            print('%-30s %6g %10.4f %10s %9s' % (k, x['lr'], x['validation'],
                                                 ('%.1f' % x['repair']) if x['repair'] is not None else '-',
                                                 ('%.3f' % x['hold_rate']) if x['hold_rate'] is not None else '-'))
        print()
        print('ogrenen kosul sayisi (kol@lr):', r['learned_by_arm_and_lr'])
        print('written', Path(args.out, 'lr-report.json'))
        return
    m = verify_frozen(args.out)
    wanted = [c for c in conditions(m['protocol']) if args.only in (None, ident(c))]
    if not wanted:
        raise SystemExit('no condition named %r' % args.only)
    for c in wanted:
        r = run_one(args.out, c)
        print(ident(c), 'val %.4f' % r['selected_validation']['final_accuracy'], '| %.0fs' % r['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
