"""E-S-PE (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.52): the E-S transformer baseline WITH positions.

E-S's transformer arm is position-free by design (depth_models.TfmArm, NoPE, four layers), and the E-RM toy showed that a
position-free encoder is handicapped at retrieval that rests on adjacency (the full-context control went from 0.265 to 0.9994 when
positions were added). So the sk4 advantage of the feedback Core over "the transformer" holds only for a NoPE transformer until
this arm is run: `tfm_pe` = the same TfmArm (same seed, same weights) with fixed sinusoidal absolute positions added to its token
embedding (parameter-free; horizon.erm_r.Positional). Training, evaluation and the reading are sk_run's, unchanged.

  python -B -m horizon.sk_pe --task sk4 --width 64 --seed 1 --lr 0.001 --steps 15000 --out results/ES-pe/sk4-tfm_pe-w64-s1-lr0.001.json
"""
import argparse
import json

import torch

from . import depth_models as dm
from . import sk_run as sr
from .erm_r import Positional

MAX_LEN = 4096          # longer than every E-S sequence (walk 512 evaluates at about 690 positions)


def make_arm(arm, task, width, seed, **kw):
    if arm == 'tfm_pe':
        model = dm.make_arm('tfm', task, width, seed, **kw)
        model.embedding = Positional(model.embedding, max_len=MAX_LEN)
        return model
    return dm.make_arm(arm, task, width, seed, **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', default='sk4')
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--log-every', type=int, default=1000)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    sr.make_arm = make_arm                       # sk_run builds its model through this name; only `tfm_pe` is new
    r = sr.run(a.task, 'tfm_pe', a.width, a.seed, a.steps, out=a.out, batch=64, lr=a.lr, lr_decay_fraction=0.3, log_every=a.log_every)
    print(json.dumps({k: r[k] for k in ('task', 'arm', 'width', 'seed', 'in_range', 'extrapolation', 'state_extrapolation', 'wall_seconds')}))


if __name__ == '__main__':
    main()
