"""Diagnose a flat needle zero: was the format never learned, or learned and not generalised?

The M0 probe finished 5000 steps (81.9 MB) with validation bits-per-byte down from 2.267 to 1.655 and exact_match
0.000 in every one of its 76 needle measurements - including at filler 0, where the fact sits immediately before
the question. The scorer reaches 1.000 against a perfect oracle, so the zeros are real.

A zero on the evaluation sets has two causes that call for opposite decisions:

    the format is not in the model at all        -> budget or task design; M1 would measure nothing
    the format is there on the keys and distances
    training actually showed, and fails outside   -> a generalisation result, which is what the gate is about

The evaluation draws keys from EVAL_KEYS, disjoint from the TRAIN_KEYS mixed into training, and fillers beyond the
0-64 range training used. So the two hypotheses are separated by scoring the SAME layout on the training
distribution: TRAIN_KEYS at a filler training saw. Anything the model learned should show up there first.

Development diagnostic. Reads a checkpoint, writes a JSON beside it, trains nothing and freezes nothing.

Usage: python -B -m mline.diagnose --checkpoint results/M0-emergence/latest.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from horizon import sweep as sw
from mline import data as md
from mline import lm as ml
from mline import needle as nd
from mline import train as mt

ROLE = 'development diagnostic: separates "format never learned" from "learned but not generalised"'
FILLERS = (0, 16, 32, 64, 128, 512)


def run(checkpoint, out=None, arm='plain', device='cuda', items=128, batch=16, seed=117):
    p = mt.protocol()
    model = ml.build(arm, width=p['width'], heads=p['heads'], window=p['window'], alpha=p['alpha'],
                     seed=seed, gate_k=p['gate_k']).to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    corpus = md.load(p['data'], 'validation')
    forward = lambda xb: ml.forward_logits(model, xb)
    rows = []
    for name, letters in (('TRAIN_KEYS', nd.TRAIN_KEYS), ('EVAL_KEYS', nd.EVAL_KEYS)):
        for n in FILLERS:
            trained_on = (name == 'TRAIN_KEYS' and n <= max(p['needle_train_filler']))
            sets = nd.make(items, n_filler=n, kind='repeat', seed=2024 + n, alphabet=letters)
            score = nd.score(forward, sets, batch=batch, device=device)
            rows.append(dict(alphabet=name, n_filler=n, width=sets['width'], in_training_distribution=trained_on,
                             exact_match=score['exact_match'], byte_accuracy=score['byte_accuracy']))
            print('%-11s filler %4d  width %5d  %-16s exact %.4f  byte %.4f'
                  % (name, n, sets['width'], '(training range)' if trained_on else '', score['exact_match'],
                     score['byte_accuracy']), flush=True)
    report = dict(kind='M0_DIAGNOSIS', role=ROLE, utc=time.time(), checkpoint=str(checkpoint),
                  checkpoint_sha256=sw.sha(checkpoint), arm=arm, items=items, rows=rows,
                  digit_chance=1 / len(nd.VALUE_ALPHABET), exact_chance=(1 / len(nd.VALUE_ALPHABET)) ** 4,
                  reading='TRAIN_KEYS inside the 0-64 filler range is the training distribution itself. If the '
                          'model scores at chance there, the needle format is absent from the model and no '
                          'comparison of arms at this budget can say anything about the gate.')
    out = Path(out or Path(checkpoint).parent / 'diagnosis.json')
    sw.write_json(out, report)
    print('\nwritten', out)
    return report


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='results/M0-emergence/latest.pt')
    parser.add_argument('--out', default=None); parser.add_argument('--arm', default='plain')
    parser.add_argument('--device', default='cuda'); parser.add_argument('--items', type=int, default=128)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    run(args.checkpoint, args.out, args.arm, args.device, args.items, args.batch)


if __name__ == '__main__':
    main()
