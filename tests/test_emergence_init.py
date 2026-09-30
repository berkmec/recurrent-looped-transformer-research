"""A fine-tune that starts anywhere but the saved weights is a different experiment, so the loader is pinned.

Two formats exist in this repository: a BARE state dict (`selected.pt`, written when a run finishes) and a training
checkpoint PAYLOAD (`latest.pt`: format, step, meta, model, optimizer). Both must load exactly, and a checkpoint
that does not match the model must fail loudly rather than half-load.
"""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from core.model import Core
from horizon import sweep as sw
from mline import checkpoint as ck
from mline import emergence as em


def tiny(seed, width=32):
    torch.manual_seed(seed)
    core = Core(width=width, heads=2, window=4, classes=256, feedback=True, alpha=.1)
    core.embedding = nn.Embedding(256, width)
    return core


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


class LoadInitTests(unittest.TestCase):
    def test_a_bare_state_dict_loads_exactly(self):
        src, dst = tiny(1), tiny(2)
        self.assertFalse(same(src, dst))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'selected.pt'
            torch.save(src.state_dict(), path)
            digest = em.load_init(dst, path)
            self.assertEqual(digest, sw.sha(path))
        self.assertTrue(same(src, dst))

    def test_a_training_checkpoint_payload_loads_exactly(self):
        src, dst = tiny(1), tiny(2)
        opt = torch.optim.AdamW(src.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ck.LATEST
            ck.save(path, src, opt, step=7, meta=dict(note='x'))
            em.load_init(dst, path)
        self.assertTrue(same(src, dst))

    def test_a_mismatched_checkpoint_fails_loudly_and_changes_nothing(self):
        src, dst = tiny(1, width=32), tiny(2, width=64)
        before = {k: v.clone() for k, v in dst.state_dict().items()}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'selected.pt'
            torch.save(src.state_dict(), path)
            with self.assertRaises(RuntimeError):
                em.load_init(dst, path)
        for k, v in dst.state_dict().items():
            self.assertTrue(torch.equal(before[k], v), k)


class FlagTests(unittest.TestCase):
    def test_the_distance_range_flag_parses_to_a_pair_of_ints(self):
        self.assertEqual(em.parse_range('0,384'), (0, 384))
        with self.assertRaises(ValueError):
            em.parse_range('384,0')
        with self.assertRaises(ValueError):
            em.parse_range('0,1,2')


if __name__ == '__main__':
    unittest.main()
