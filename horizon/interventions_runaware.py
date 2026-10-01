"""U2b: run-aware freeze. Revert the decoder state (and repeat the logits) only from the (k+1)-th consecutive neutral token.

Motivation: U2-measure shows zero displacement after k <= 2 neutrals, while the plain freeze arm (U2-intervene) costs
~4 pp on clean sequences because token 2 is a natural op. Freezing later should keep the clean cost near zero.
Arms: none (plain decode, from interventions.run), freeze_k1, freeze_k2. Oracle neutral knowledge (ceiling), no training.
"""
import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from horizon import interventions as iv
from horizon import metrics as mt
from horizon import sweep as sw
from horizon import task_families as tf

ROOT = Path(__file__).resolve().parents[1]
SOURCES = iv.SOURCES + ('horizon/interventions_runaware.py',)
ARMS = ('none', 'freeze_k1', 'freeze_k2')


def run_length(x):
    counts = torch.zeros_like(x); run = torch.zeros(x.shape[0], dtype=x.dtype)
    for t in range(x.shape[1]):
        run = torch.where(x[:, t] == iv.NEUTRAL, run + 1, torch.zeros_like(run)); counts[:, t] = run
    return counts


@torch.no_grad()
def run(model, x, k=2):
    model.eval()
    e, _, _ = model.encode(x)
    n = model.mn(e)
    mk = model.rope(model.split(model.mk(n))); mv = model.split(model.mv(n))
    lengths = run_length(x)
    state, prev_logits = None, None
    logits, states = [], []
    for t in range(x.shape[1]):
        frozen = lengths[:, t] > k
        z, new_state = model.decode(e[:, t], state, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
        if state is not None and bool(frozen.any()):
            m = frozen.view(-1, 1)
            s = torch.where(m, state['s'], new_state['s'])
            dk, dv = new_state['dk'], new_state['dv']
            if state['dk'].shape == dk.shape:
                m4 = frozen.view(-1, 1, 1, 1)
                dk = torch.where(m4, state['dk'], dk); dv = torch.where(m4, state['dv'], dv)
            new_state = dict(new_state, s=s, dk=dk, dv=dv)
            z = torch.where(m, prev_logits, z)
        state, prev_logits = new_state, z
        logits.append(z); states.append(state['s'])
    return torch.stack(logits, 1), torch.stack(states, 1)


def run_arm(model, x, arm, batch=32):
    logits = []
    for at in range(0, len(x), batch):
        xb = x[at:at + batch]
        z = iv.run(model, xb, 'none')[0] if arm == 'none' else run(model, xb, k=int(arm.split('_k')[1]))[0]
        logits.append(z)
    return torch.cat(logits)


def protocol():
    p = iv.protocol()
    p.update(purpose='U2b run-aware freeze at neutral positions: revert only from the (k+1)-th consecutive neutral (ceiling with oracle neutral knowledge)',
             arms=dict(five=list(ARMS), stack=list(ARMS)))
    return p


def source_hashes():
    return {name: iv.sha(ROOT / name) for name in SOURCES}


def freeze(out, p=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(p or protocol())
    manifest = dict(protocol=p, protocol_hash=iv.digest(p), sources=source_hashes(), u1_protocol_hash=sw.verify_frozen(p['u1'])['protocol_hash'],
                    environment=dict(python=platform.python_version(), torch=torch.__version__), frozen_utc=time.time())
    if (out / 'frozen.json').exists():
        old = verify_frozen(out)
        if old['protocol_hash'] != manifest['protocol_hash']:
            raise ValueError('refusing frozen protocol change')
        return old
    if any(out.iterdir()):
        raise ValueError('new freeze requires empty output directory')
    iv.write_json(out / 'frozen.json', manifest)
    (out / 'frozen.sha256').write_text(iv.sha(out / 'frozen.json'), encoding='ascii')
    return manifest


def verify_frozen(out):
    out = Path(out)
    if iv.sha(out / 'frozen.json') != (out / 'frozen.sha256').read_text(encoding='ascii').strip():
        raise ValueError('frozen manifest hash mismatch')
    manifest = json.loads((out / 'frozen.json').read_text(encoding='utf8'))
    if manifest['protocol_hash'] != iv.digest(manifest['protocol']) or manifest['sources'] != source_hashes():
        raise ValueError('protocol/source drift')
    torch.set_num_threads(1)
    return manifest


def run_condition(out, cid):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    path = out / f'{cid}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    m1 = sw.verify_frozen(p['u1']); p1 = m1['protocol']; c = sw.known(m1, cid)
    model = sw.make_model(c, p1); model.load_state_dict(torch.load(Path(p['u1']) / cid / 'selected.pt', weights_only=True))
    t0 = time.perf_counter()
    result = dict(condition=cid, task=c['task'], seed=c['seed'], selected_sha256=iv.sha(Path(p['u1']) / cid / 'selected.pt'), protocol_hash=manifest['protocol_hash'], arms={})
    x, y, seed = tf.heldout(p1['source_seeds'], c['task'], c['seed'], 'neutral', p['n_neutral'], p1['train_length'])
    clean = {L: tf.heldout(p1['source_seeds'], c['task'], c['seed'], 'test', p['n_clean'], L) for L in p['clean_lengths']}
    for arm in p['arms'][c['task']]:
        entry = dict(neutral={}, clean={})
        base_pred = run_arm(model, x, arm).argmax(-1)[:, -1]
        entry['neutral']['0-base'] = dict(final_accuracy=float((base_pred == y[:, -1]).float().mean()))
        for slots in p['neutral_slots']:
            for placement in p['placements']:
                xx, yy = tf.insert_neutrals(x, c['task'], slots, placement)
                pred = run_arm(model, xx, arm).argmax(-1)[:, -1]
                entry['neutral'][f'{slots}-{placement}'] = dict(final_accuracy=float((pred == yy[:, -1]).float().mean()), **mt.flip_counts(base_pred.numpy(), pred.numpy(), yy[:, -1].numpy()))
        for L, (cx, cy, cseed) in clean.items():
            lg = run_arm(model, cx, arm)
            entry['clean'][str(L)] = dict(final_accuracy=float((lg.argmax(-1)[:, -1] == cy[:, -1]).float().mean()), prefix_accuracy=float((lg.argmax(-1) == cy).float().mean()), source_seed=cseed)
        result['arms'][arm] = entry
    result['seconds'] = time.perf_counter() - t0
    iv.write_json(path, result)
    return result


def analyze(out):
    out = Path(out); manifest = verify_frozen(out); p = manifest['protocol']
    m1 = sw.verify_frozen(p['u1']); cs = sw.conditions(m1['protocol'])
    rows = {sw.ident(c): run_condition(out, sw.ident(c)) for c in cs}
    summary = {}
    for task in ('five', 'stack'):
        seeds = [r for r in rows.values() if r['task'] == task]
        for arm in p['arms'][task]:
            if arm == 'none':
                continue
            entry = dict(neutral={}, clean={})
            for key in seeds[0]['arms'][arm]['neutral']:
                if key == '0-base':
                    continue
                diffs = [(r['arms'][arm]['neutral'][key]['final_accuracy'] - r['arms']['none']['neutral'][key]['final_accuracy']) * 100 for r in seeds]
                loss = [(r['arms'][arm]['neutral']['0-base']['final_accuracy'] - r['arms'][arm]['neutral'][key]['final_accuracy']) * 100 for r in seeds]
                entry['neutral'][key] = dict(gain_vs_none_pp=mt.seed_summary(diffs), loss_vs_own_base_pp=mt.seed_summary(loss), within_target=[l <= p['targets']['neutral_loss_pp'] for l in loss])
            for L in p['clean_lengths']:
                diffs = [(r['arms'][arm]['clean'][str(L)]['final_accuracy'] - r['arms']['none']['clean'][str(L)]['final_accuracy']) * 100 for r in seeds]
                entry['clean'][str(L)] = dict(diff_vs_none_pp=mt.seed_summary(diffs), no_regression=[d >= -p['targets']['clean_regression_pp'] for d in diffs])
            all_neutral = all(all(v['within_target']) for v in entry['neutral'].values())
            all_clean = all(all(v['no_regression']) for v in entry['clean'].values())
            entry['status'] = 'MEASURED_POSITIVE_CEILING' if all_neutral and all_clean else 'MIXED' if any(any(v['within_target']) for v in entry['neutral'].values()) else 'MEASURED_NEGATIVE'
            summary[f'{task}/{arm}'] = entry
    result = dict(status='MEASURED', protocol_hash=manifest['protocol_hash'], u1_protocol_hash=manifest['u1_protocol_hash'], conditions=rows, summary=summary,
                  caveat='oracle neutral-token knowledge; ceiling for a learned no-op detector; three seeds; exploratory intervals')
    iv.write_json(out / 'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--run-all', action='store_true'); group.add_argument('--analyze', action='store_true')
    parser.add_argument('--out', default='results/U2b-intervene')
    args = parser.parse_args()
    if args.freeze:
        result = dict(protocol_hash=freeze(args.out)['protocol_hash'])
    elif args.run_all:
        m = verify_frozen(args.out); m1 = sw.verify_frozen(m['protocol']['u1'])
        result = {}
        for c in sw.conditions(m1['protocol']):
            r = run_condition(args.out, sw.ident(c)); result[sw.ident(c)] = {arm: {k: round(v['final_accuracy'], 3) for k, v in e['neutral'].items()} for arm, e in r['arms'].items()}
            print(sw.ident(c), json.dumps(result[sw.ident(c)]), f"{r['seconds']:.0f}s", flush=True)
    else:
        a = analyze(args.out); result = {k: dict(status=v['status'], neutral={kk: round(vv['loss_vs_own_base_pp']['mean'], 1) for kk, vv in v['neutral'].items()}, clean={kk: round(vv['diff_vs_none_pp']['mean'], 2) for kk, vv in v['clean'].items()}) for k, v in a['summary'].items()}
    print(json.dumps(result, indent=1, default=str))


if __name__ == '__main__':
    main()
