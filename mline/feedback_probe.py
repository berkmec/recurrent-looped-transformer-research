"""E-L1P (R5-E1): does the L1 model USE its feedback channel for language? Evaluation only, CPU, on the snapshots the run saves every 1,000 updates.

The channel is the previous final decoder output fed back through the gated merge (alpha 0.1 in training). Re-evaluating the SAME validation batches the L1
battery uses (seed 991, 8 batches of 32 x 512 bytes) with the channel changed shows how much the model leans on it:

  normal       alpha 0.1, as trained (must reproduce the bits per byte the run recorded at that update, to 0.003)
  a0           alpha 0: the channel is cut at inference
  a005 / a02   alpha 0.05 / 0.2
  zero_state   the fed-back state is replaced by zeros (the merge then sees an all-zero r). NOTE (review finding F5): RMSNorm(0) = 0 and the merge has no bias, so this
               is the same function as `a0` wherever the state is replaced; the two differ only at the first step (where the learned `initial` vector still feeds the
               channel) and, through the decoder's 3-step window cache, at positions 1-3: it is a consistency check, NOT independent evidence that the channel
               carries content
  other_doc    the fed-back state is the one of another sequence of the batch (rolled by one): a state that carries nothing about THIS text

Reported: bits per byte overall and in position slices [0, 64) [64, 256) [256, 512), and the difference to `normal`. A model that never learned to use the channel
shows ~0 everywhere (a plain transformer with a few thousand extra parameters, the reading R5 gives for the case of an ignored channel); a model that does shows a
positive difference that grows with the position. Reading rule (program note section 8): `channel used for language` if a0 - normal >= 0.01 bits per byte
overall; `not used` if a0 - normal < 0.002 AND other_doc - normal < 0.002; in between: reported without a label. Cutting a channel the model relies on also moves
the input distribution, so the difference measures reliance, not benefit; benefit needs the A0 twin (same data, same order, no feedback), which does not exist yet.

  python -B -m mline.feedback_probe results/L1-scratch-w928-s401/snapshots/step002000.pt --out results/L1-feedback-probe/step002000-feedback-probe.json
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from core.model import Core
from mline import data as md
from mline import lm as ml
from mline import train as mt

CONDITIONS = ('normal', 'a0', 'a005', 'a02', 'zero_state', 'other_doc')
ALPHA = dict(normal=0.1, a0=0.0, a005=0.05, a02=0.2, zero_state=0.1, other_doc=0.1)
SLICES = ((0, 64), (64, 256), (256, 512))


class SubstitutedCore(Core):
    """The sealed Core whose fed-back state (the previous final decoder output) is replaced before every decoder step but the first."""

    mode = None

    def decode(self, e, s, mk, mv, t):
        if s is not None and self.mode:
            s = dict(s)
            s['s'] = torch.zeros_like(s['s']) if self.mode == 'zero' else s['s'].roll(1, dims=0)
        return super().decode(e, s, mk, mv, t)


def configure(model, name):
    """Set the channel condition in place (class swap + alpha); `normal` restores the trained behaviour."""
    if name not in CONDITIONS:
        raise ValueError('unknown condition: ' + str(name))
    model.alpha = ALPHA[name]
    model.__class__ = SubstitutedCore
    model.mode = {'zero_state': 'zero', 'other_doc': 'other'}.get(name)
    return model


def load_model(snapshot, width=928, heads=8, seed=401):
    model = ml.build('plain', width=width, heads=heads, window=4, alpha=0.1, seed=seed)
    model.load_state_dict(torch.load(snapshot, weights_only=True))
    return model.eval()


def recorded_bits_per_byte(snapshot, log):
    """The validation bits per byte the run itself recorded at the snapshot's update (None when the log has no such row)."""
    match = re.search(r'step(\d+)\.pt$', str(snapshot))
    if not match or not Path(log).exists():
        return None
    step = int(match.group(1))
    for line in Path(log).read_text(encoding='utf8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('step') == step and isinstance(row.get('eval'), dict):
                return row['eval'].get('bits_per_byte')
    return None


def per_position_loss(model, x, y):
    with torch.no_grad():
        logits = model(x)
    return F.cross_entropy(logits.transpose(1, 2), y, reduction='none')


def evaluate(model, batches, conditions=CONDITIONS, slices=SLICES, log=print):
    """{condition: {bits_per_byte, slices}} over the same (x, y) batches for every condition."""
    out = {}
    for name in conditions:
        configure(model, name)
        started, total, n = time.perf_counter(), None, 0
        for x, y in batches:
            ce = per_position_loss(model, torch.from_numpy(x), torch.from_numpy(y))
            total = ce.sum(0) if total is None else total + ce.sum(0)
            n += ce.shape[0]
        per_position = total / n / math.log(2)
        out[name] = dict(bits_per_byte=float(per_position.mean()), slices={f'{a}-{b}': float(per_position[a:b].mean()) for a, b in slices if a < per_position.shape[0]},
                         seconds=time.perf_counter() - started)
        log(f"  {name:<10} {out[name]['bits_per_byte']:.4f} b/B  " + '  '.join(f'[{k.replace("-", ",")}) {v:.4f}' for k, v in out[name]['slices'].items())
            + f"  ({out[name]['seconds']:.0f} s)")
    return out


def reading(results, recorded=None, tolerance=0.003):
    """The rule written before the probe ran (see the module docstring)."""
    base = results['normal']['bits_per_byte']
    diff = {n: r['bits_per_byte'] - base for n, r in results.items() if n != 'normal'}
    valid = None if recorded is None else abs(base - recorded) <= tolerance
    a0, other = diff.get('a0'), diff.get('other_doc')
    if valid is False:
        verdict = 'INVALID: the normal row does not reproduce the recorded validation'
    elif valid is None:      # nothing to reproduce (the step is missing from the log, or the battery wrote an error): the rule says the snapshot is not read (review finding F9)
        verdict = 'NOT READ: no recorded validation to reproduce'
    elif a0 is not None and a0 >= 0.01:
        verdict = 'CHANNEL USED FOR LANGUAGE'
    elif a0 is not None and other is not None and a0 < 0.002 and other < 0.002:
        verdict = 'NOT USED'
    else:
        verdict = 'IN BETWEEN'
    return dict(verdict=verdict, differences=diff, normal_reproduces_recorded=valid, recorded=recorded)


def run(snapshot, directory=None, batches=8, batch=32, length=512, width=928, heads=8, log=print):
    protocol = mt.protocol()
    data = directory or protocol['data']
    val = md.load(data, 'validation')
    pairs = [md.batch(val, 991, step, batch, length)[:2] for step in range(1, batches + 1)]
    model = load_model(snapshot, width=width, heads=heads)
    results = evaluate(model, pairs, log=log)
    recorded = recorded_bits_per_byte(snapshot, Path(snapshot).parent.parent / 'probe.jsonl')
    return dict(snapshot=str(snapshot), batches=batches, batch=batch, length=length, results=results, reading=reading(results, recorded))


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('snapshots', nargs='+')
    p.add_argument('--batches', type=int, default=8)
    p.add_argument('--out', default=None, help='JSON file (one snapshot) or directory (several)')
    a = p.parse_args()
    for path in a.snapshots:
        print(path, flush=True)
        r = run(path, batches=a.batches, log=lambda m: print(m, flush=True))
        print('  reading:', json.dumps(r['reading'], default=float), flush=True)
        if a.out:
            target = Path(a.out)
            if len(a.snapshots) > 1 or target.is_dir():
                target = target / (Path(path).stem + '-feedback-probe.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(r, indent=1, default=float), encoding='utf8')


if __name__ == '__main__':
    main()
