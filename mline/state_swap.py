"""State swap on a saved checkpoint: whose answer comes out when the carried state belongs to another item?

The plain Core carries a recurrent state s_t from step to step (a weak additive feedback, but one the model demonstrably
depends on: switching it off destroys retrieval beyond ~40 bytes, results/M1-diagnosis/channel-plain.json) next to full
causal attention. Switching the feedback OFF is not a clean cut, because it changes the function the weights were trained
for. A swap is: at ONE step, row i receives row donor[i]'s state instead of its own, and everything else is untouched. The
next digit then tells us where the fact was read from:

    it follows the DONOR      the fact was carried in the recurrent state
    it follows the RECIPIENT  the fact came from attention (the row's own encoder memory)
    neither                   the swap broke the model, or the answer needed both

Readings are taken only on rows where donor and recipient differ in the first answer digit, so the two cannot be mixed up.
Chance for each outcome is 0.1. The forward pass is the sealed Core.forward loop with one extra line; with no swap it is
bitwise the sealed forward (tests/test_state_swap.py).

Development diagnostic. Reads a checkpoint, trains nothing, freezes nothing. Plain arm only: the gated arm's gate never
closed in M1, so it computes the same function.

Usage: python -B -m mline.state_swap --checkpoint <selected.pt> --out <json>
"""
import argparse
import sys
import time
from pathlib import Path

import torch

from horizon import sweep as sw
from mline import emergence as em
from mline import lm as ml
from mline import needle as nd
from mline import train as mt

ROLE = 'development diagnostic: is the needle read from the recurrent state or from attention?'
FACT_BYTES = 27                       # the fact sentence plus its newline: 'The code for KKKK is VVVV.' + newline
POINTS = ('fact', 'question', 'answer')


def forward_with_swap(core, x, swap_at=None, donor=None):
    """Core.forward's recurrent loop; at step swap_at each row takes the carried state of row donor[i]."""
    if swap_at is not None and donor is None:
        raise ValueError('a swap needs a donor permutation')
    e, _, _ = core.encode(x)
    n = core.mn(e)
    mk = core.rope(core.split(core.mk(n)))
    mv = core.split(core.mv(n))
    state, out = None, []
    for t in range(x.shape[1]):
        if swap_at is not None and t == swap_at and state is not None:
            state = dict(state)
            state['s'] = state['s'][donor]
        y, state = core.decode(e[:, t], state, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
        out.append(y)
    return torch.stack(out, 1)


def clean_states(core, x):
    """The carried state after each step of the ordinary forward pass: [batch, steps, width]."""
    e, _, _ = core.encode(x)
    n = core.mn(e)
    mk = core.rope(core.split(core.mk(n)))
    mv = core.split(core.mv(n))
    state, states = None, []
    for t in range(x.shape[1]):
        _, state = core.decode(e[:, t], state, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
        states.append(state['s'])
    return torch.stack(states, 1)


def forward_with_transplant(core, x, from_step, source):
    """From step `from_step` on, the feedback each row receives is source[:, t - 1] instead of its own carried state.

    `source` is a [batch, steps, width] stream: another row's clean states (a donor) or their batch mean (generic
    content with the item-specific part averaged out). Everything else - the row's own encoder, attention and caches -
    is untouched, so whatever the answer reads from the recurrent channel after `from_step` has to come from `source`.
    """
    e, _, _ = core.encode(x)
    n = core.mn(e)
    mk = core.rope(core.split(core.mk(n)))
    mv = core.split(core.mv(n))
    state, out = None, []
    for t in range(x.shape[1]):
        if t >= from_step and state is not None:
            state = dict(state)
            state['s'] = source[:, t - 1]
        y, state = core.decode(e[:, t], state, mk[:, :, :t + 1], mv[:, :, :t + 1], t)
        out.append(y)
    return torch.stack(out, 1)


KINDS = ('swap', 'donor', 'mean')


def parse_condition(label):
    """'none', or '<kind>-<point>' with kind in swap / donor / mean and point in fact / question / answer."""
    if label == 'none':
        return 'none', None
    parts = label.split('-')
    if len(parts) != 2 or parts[0] not in KINDS or parts[1] not in POINTS:
        raise ValueError('a condition is none or KIND-POINT, KIND in %s, POINT in %s' % (KINDS, POINTS))
    return parts[0], parts[1]


def attribute(got, own, don):
    """Share of usable rows whose first answer digit matches the donor, the recipient, or neither."""
    usable = own != don
    n = int(usable.sum())
    if n == 0:
        return dict(usable=0, donor=None, recipient=None, neither=None)
    g, o, d = got[usable], own[usable], don[usable]
    donor = float((g == d).float().mean())
    recipient = float((g == o).float().mean())
    return dict(usable=n, donor=donor, recipient=recipient, neither=1. - donor - recipient)


def swap_step(point, n_filler, width, value_length=4):
    if point == 'fact':
        return FACT_BYTES                        # right after the fact, before the filler
    if point == 'question':
        return FACT_BYTES + n_filler             # right after the filler, before the question
    if point == 'answer':
        return width - value_length - 1          # the step that predicts the first answer digit
    raise ValueError('point must be one of ' + ', '.join(POINTS))


def fmt(value):
    return '-' if value is None else '%.2f' % value


def run(checkpoint, out, fillers, conditions=('none', 'swap-fact', 'swap-question'),
        alphabets=('TRAIN_KEYS', 'EVAL_KEYS'), items=64,
        batch=16, device='cpu', seed=117):
    p = mt.protocol()
    model = ml.build('plain', width=p['width'], heads=p['heads'], window=p['window'], alpha=p['alpha'], seed=seed,
                     gate_k=p['gate_k']).to(device)
    em.load_init(model, checkpoint)
    model.eval()
    keys = dict(TRAIN_KEYS=nd.TRAIN_KEYS, EVAL_KEYS=nd.EVAL_KEYS)
    rows = []
    for name in alphabets:
        for n in fillers:
            sets = nd.make(items, n_filler=n, kind='repeat', seed=2024 + n, alphabet=keys[name])
            x, y, mask, width = sets['x'], sets['y'], sets['mask'], sets['width']
            first = width - len(sets['answers'][0]) - 1      # the position that predicts the first answer digit
            for label in conditions:
                kind, point = parse_condition(label)
                at = None if point is None else swap_step(point, n, width)
                got, own, don, exact = [], [], [], []
                for start in range(0, len(x), batch):
                    xb, yb, mb = x[start:start + batch], y[start:start + batch], mask[start:start + batch]
                    xd = xb.to(device)
                    donor = torch.roll(torch.arange(len(xb)), 1)
                    with torch.no_grad():
                        if kind == 'none':
                            logits = forward_with_swap(model, xd)
                        elif kind == 'swap':
                            logits = forward_with_swap(model, xd, at, donor.to(device))
                        else:
                            clean = clean_states(model, xd)
                            if kind == 'donor':
                                source = clean[donor.to(device)]
                            else:
                                source = clean.mean(0, keepdim=True).expand_as(clean)
                            logits = forward_with_transplant(model, xd, at, source)
                    pred = logits.argmax(-1).cpu()
                    got.append(pred[:, first])
                    own.append(yb[:, first])
                    don.append(yb[donor, first])
                    exact.append(((pred == yb) & mb).sum(1) == mb.sum(1))
                got, own, don, exact = torch.cat(got), torch.cat(own), torch.cat(don), torch.cat(exact)
                row = dict(alphabet=name, n_filler=n, width=width, condition=label, swap_at=at,
                           exact_match_own=float(exact.float().mean()),
                           first_digit_own=float((got == own).float().mean()))
                row.update(attribute(got, own, don))
                if kind == 'mean':
                    row.update(donor=None, neither=None)     # no donor exists: only the recipient's own digit is read
                rows.append(row)
                print('%-10s filler %4d  %-8s exact(own) %.2f  first-digit own %.2f | usable %3d  donor %s  '
                      'recipient %s  neither %s' % (name, n, label, row['exact_match_own'], row['first_digit_own'],
                                                    row['usable'], fmt(row['donor']), fmt(row['recipient']),
                                                    fmt(row['neither'])), flush=True)
    report = dict(kind='M1_STATE_SWAP', role=ROLE, utc=time.time(), checkpoint=str(checkpoint),
                  checkpoint_sha256=sw.sha(checkpoint), items=items, rows=rows,
                  reading='donor high: the fact travels in the recurrent state. recipient high: attention carries it. '
                          'Chance for each is 0.1; only rows whose first answer digit differs between donor and '
                          'recipient are counted.')
    sw.write_json(Path(out), report)
    print('written', out)
    return report


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fillers', default='0,32,48,64')
    parser.add_argument('--conditions', default='none,swap-fact,swap-question',
                        help='none, or KIND-POINT: swap (one step), donor (whole donor stream), mean (batch mean)')
    parser.add_argument('--items', type=int, default=64)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(1)                 # a CPU worker shares the machine with a training run
    run(args.checkpoint, args.out, [int(v) for v in args.fillers.split(',')], tuple(args.conditions.split(',')),
        items=args.items, batch=args.batch, device=args.device)


if __name__ == '__main__':
    main()
