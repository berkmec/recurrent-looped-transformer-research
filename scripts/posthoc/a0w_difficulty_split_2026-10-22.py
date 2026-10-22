"""POST-HOC (not pre-registered) check of the E-A0W difficulty split: binning by A0's own loss favours L1 by regression to the mean,
so bin also by L1's own loss (the opposite bias) and by the mean of the two (the fair split). Same bytes and snapshots as E-A0W."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
torch.cuda.set_per_process_memory_fraction(0.25)
from mline import a0_where as aw  # noqa: E402
from mline import data as md  # noqa: E402
from mline import lm as ml  # noqa: E402
from mline import train as mt  # noqa: E402

R = 'results/'
val = md.load(mt.protocol()['data'], 'validation')
xs, ys = zip(*[md.batch(val, 401, step, 8, 512)[:2] for step in range(16)])
x, y = np.concatenate(xs), np.concatenate(ys)
tx, ty = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
loss = {}
for arm, path in (('a0', R + 'L1-a0-w928-s401/snapshots/step008000.pt'), ('plain', R + 'L1-scratch-w928-s401/snapshots/step008000.pt')):
    model = ml.build(arm, width=928, heads=8, seed=401)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    loss[arm] = aw.per_byte_loss(model.cuda().eval(), tx, ty) / np.log(2)
    del model
a0, l1 = loss['a0'], loss['plain']
delta = a0 - l1
out = dict(note='POST-HOC, not pre-registered', gain_bits_per_byte=float(delta.mean()))
names = ['easiest 25 %', '25-50 %', '50-75 %', 'hardest 25 %']
for key, ref in (('by_a0_quartile', a0), ('by_l1_quartile', l1), ('by_mean_quartile', (a0 + l1) / 2)):
    groups = np.digitize(ref, np.quantile(ref, (0.25, 0.5, 0.75)))
    out[key] = aw.split(delta, a0, groups, names)

text = json.dumps(out, indent=1)
open(R + 'L1-a0-w928-s401/where-step008000-posthoc.json', 'wb').write((text + '\n').encode('utf8'))
for key in ('by_a0_quartile', 'by_l1_quartile', 'by_mean_quartile'):
    print(key, {k: (round(v['gain_bits_per_byte'], 4), round(v['share_of_gain'], 3)) for k, v in out[key].items()})
