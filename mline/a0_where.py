"""E-A0W (notes/RLT_SORUN_ESLEME_VE_PROGRAM_2026-10-02.md section 5.45; descriptive, no verdict): where does the feedback earn its
+0.05 bit/byte? The per-byte loss of L1 (feedback on) and of its A0 twin (feedback off) at the same update, on the same validation
bytes, split by position in the 512-byte window, by the class of the predicted byte, by how far into a word the predicted letter
is, and by how hard the byte is for A0.

Why these splits: the window-4 decoder self-attention sees the PRE-attention states of the last positions; only the feedback
brings position t the FINAL state of t - 1 (after cross-attention and the feed-forward), and through it every earlier final
state. So the feedback adds sequential depth over a one-layer encoder and a one-layer decoder, not context (the encoder already
sees the whole prefix). Depth should matter most where bytes have to be composed (inside words), less at word starts.

  python -B -m mline.a0_where --l1 results/L1-scratch-w928-s401/snapshots/step008000.pt \
      --a0 results/L1-a0-w928-s401/snapshots/step008000.pt --out results/L1-a0-w928-s401/where-step008000.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

POSITION_BINS = ((0, 4), (4, 16), (16, 64), (64, 256), (256, 512))
CLASSES = ('space', 'newline', 'letter', 'digit', 'punctuation', 'utf8-lead', 'utf8-continuation', 'other')


def byte_class(b):
    """Class index (into CLASSES) of every byte of an integer array."""
    b = np.asarray(b)
    out = np.full(b.shape, CLASSES.index('other'), dtype=np.int64)
    out[(b >= 33) & (b <= 126)] = CLASSES.index('punctuation')
    out[(b >= 48) & (b <= 57)] = CLASSES.index('digit')
    out[((b >= 65) & (b <= 90)) | ((b >= 97) & (b <= 122))] = CLASSES.index('letter')
    out[b == 32] = CLASSES.index('space')
    out[b == 10] = CLASSES.index('newline')
    out[(b >= 0x80) & (b <= 0xBF)] = CLASSES.index('utf8-continuation')
    out[(b >= 0xC0) & (b <= 0xF7)] = CLASSES.index('utf8-lead')
    return out


def word_offset(x):
    """For rows of input bytes x (the context; the target at t is x[t + 1]), the number of consecutive ASCII letters ending at
    x[t]: 0 means the byte predicted at t would be the first letter of a word."""
    x = np.asarray(x)
    letter = ((x >= 65) & (x <= 90)) | ((x >= 97) & (x <= 122))
    out = np.zeros(x.shape, dtype=np.int64)
    run = np.zeros(x.shape[0], dtype=np.int64)
    for t in range(x.shape[1]):
        run = np.where(letter[:, t], run + 1, 0)
        out[:, t] = run
    return out


def split(delta, a0, groups, names):
    """Mean gain (bits per byte) and share of the total gain per group; groups is an integer array like delta."""
    total = float(delta.sum())
    rows = {}
    for i, name in enumerate(names):
        m = groups == i
        n = int(m.sum())
        if n:
            rows[name] = dict(bytes=n, gain_bits_per_byte=float(delta[m].mean()), a0_bits_per_byte=float(a0[m].mean()),
                              share_of_gain=float(delta[m].sum() / total) if total else None)
    return rows


def analyse(ce_a0, ce_l1, x, y):
    """ce_* in nats per predicted byte, shaped like y; x the inputs (y[:, t] = x[:, t + 1] inside the corpus stream)."""
    a0, l1 = ce_a0 / math.log(2), ce_l1 / math.log(2)
    delta = a0 - l1
    T = y.shape[1]
    pos = np.zeros(y.shape, dtype=np.int64)
    for i, (lo, hi) in enumerate(POSITION_BINS):
        pos[:, lo:min(hi, T)] = i
    cls = byte_class(y)
    off = np.minimum(word_offset(x), 4)
    letter = cls == CLASSES.index('letter')
    off_groups = np.where(letter, off, -1)
    q = np.quantile(a0, (0.25, 0.5, 0.75))
    hard = np.digitize(a0, q)
    return dict(
        bytes=int(delta.size), a0_bits_per_byte=float(a0.mean()), l1_bits_per_byte=float(l1.mean()), gain_bits_per_byte=float(delta.mean()),
        gain_positive_fraction=float((delta > 0).mean()),
        by_position=split(delta, a0, pos, ['%d-%d' % b for b in POSITION_BINS]),
        by_target_class=split(delta, a0, cls, CLASSES),
        by_word_offset_of_letters=split(delta, a0, off_groups, ['0 (word start)', '1', '2', '3', '4+']),
        by_a0_difficulty_quartile=split(delta, a0, hard, ['easiest 25 %', '25-50 %', '50-75 %', 'hardest 25 %']))


def per_byte_loss(model, x, y, batch=8):
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            logits = model(x[i:i + batch])
            out.append(F.cross_entropy(logits.flatten(0, 1), y[i:i + batch].flatten(), reduction='none').view(y[i:i + batch].shape).cpu())
    return torch.cat(out).double().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--l1', required=True)
    p.add_argument('--a0', required=True)
    p.add_argument('--rows', type=int, default=128)
    p.add_argument('--length', type=int, default=512)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from mline import data as md
    from mline import lm as ml
    from mline import train as mt
    val = md.load(mt.protocol()['data'], 'validation')
    xs, ys = zip(*[md.batch(val, 401, step, 8, a.length)[:2] for step in range(a.rows // 8)])
    x, y = np.concatenate(xs), np.concatenate(ys)
    tx, ty = torch.from_numpy(x).to(a.device), torch.from_numpy(y).to(a.device)
    losses = {}
    for arm, path in (('a0', a.a0), ('plain', a.l1)):
        model = ml.build(arm, width=928, heads=8, seed=401)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        losses[arm] = per_byte_loss(model.to(a.device).eval(), tx, ty)
        del model
    r = analyse(losses['a0'], losses['plain'], x, y)
    r.update(l1=a.l1, a0=a.a0, rows=a.rows, length=a.length)
    text = json.dumps(r, indent=1)
    print(text)
    Path(a.out).write_bytes((text + '\n').encode('utf8'))


if __name__ == '__main__':
    main()
