"""A resumed M1 run must produce the SAME weights as one that was never interrupted, or it corrupts its own evidence.

Resuming is only safe here because the data pipeline is a pure function of (seed, step): `training_batch` draws
its corpus rows from `md.batch(train, seed, step, ...)` and its needles from `seed * 100_003 + step`. Nothing
carries over between steps except the model, the optimizer moments and the counters. So a run restored at step N
and continued is not an approximation of the original - it is the original, provided the restore is complete.

"Complete" is the part worth testing. Dropping AdamW's moments would silently produce a different trajectory that
still looks like a successful resume, and the run would finish with weights nobody can reproduce.

The crash is simulated the way crashes actually arrive: an exception in the middle of the loop, after some
checkpoints have been written and before the run is finished.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def tiny(data):
    from mline import train as mt
    p = dict(mt.protocol(), width=64, heads=2, steps=6, batch=4, length=64, validation_every=2,
             validation_batches=1, needle_items=4, needle_batch=2, needle_filler=[0], needle_kinds=['repeat'],
             needle_placements=['far'], arms=['plain'], seeds=[117], device='cpu', data=str(data))
    return p


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from mline import train as mt
        cls.mt = mt
        if not (Path(mt.DATA) / 'train.bin').exists():
            raise unittest.SkipTest('MISSING_ASSET: ' + mt.DATA)

    def run_condition(self, out, crash_at=None):
        mt = self.mt
        p = tiny(mt.DATA)
        mt.freeze(out, p)
        manifest = mt.verify_frozen(out)
        c = mt.conditions(manifest['protocol'])[0]
        if crash_at is None:
            return mt.run_one(out, c, device='cpu')
        real = mt.validate
        state = {'calls': 0}

        def exploding(*a, **k):
            state['calls'] += 1
            if state['calls'] >= crash_at:
                raise RuntimeError('simulated power cut')
            return real(*a, **k)
        mt.validate = exploding
        try:
            with self.assertRaises(RuntimeError):
                mt.run_one(out, c, device='cpu')
        finally:
            mt.validate = real
        return c

    def test_an_interrupted_run_resumes_to_identical_weights(self):
        mt = self.mt
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            clean = self.run_condition(Path(a))
            c = self.run_condition(Path(b), crash_at=2)          # dies at the second validation, step 4
            resumed = mt.run_one(Path(b), c, device='cpu')

            self.assertEqual(resumed['steps'], clean['steps'])
            wa = torch.load(Path(a, mt.ident(c), 'selected.pt'), weights_only=True, map_location='cpu')
            wb = torch.load(Path(b, mt.ident(c), 'selected.pt'), weights_only=True, map_location='cpu')
            self.assertEqual(sorted(wa), sorted(wb))
            for k in wa:
                self.assertTrue(torch.equal(wa[k], wb[k]), f'{k} differs after resume')
            self.assertAlmostEqual(resumed['selected_validation']['loss'],
                                   clean['selected_validation']['loss'], places=6)

    def test_the_resume_is_recorded_rather_than_silent(self):
        mt = self.mt
        with TemporaryDirectory() as b:
            c = self.run_condition(Path(b), crash_at=2)
            resumed = mt.run_one(Path(b), c, device='cpu')
            self.assertGreater(resumed['resumed_from'], 0, 'a run that was interrupted must say so in its record')

    def test_the_update_log_has_no_duplicate_or_missing_steps(self):
        mt = self.mt
        with TemporaryDirectory() as b:
            c = self.run_condition(Path(b), crash_at=2)
            mt.run_one(Path(b), c, device='cpu')
            steps = [json.loads(l)['step'] for l in
                     Path(b, mt.ident(c), 'updates.jsonl').read_text(encoding='utf8').strip().splitlines()]
            self.assertEqual(steps, list(range(1, 7)), 'the log must read as if the run was never interrupted')

    def test_a_directory_with_no_checkpoint_is_still_refused(self):
        """The old guard stays: an inconsistent directory with nothing to resume from needs a human."""
        mt = self.mt
        with TemporaryDirectory() as b:
            out = Path(b)
            p = tiny(mt.DATA)
            mt.freeze(out, p)
            c = mt.conditions(mt.verify_frozen(out)['protocol'])[0]
            directory = out / mt.ident(c); directory.mkdir(exist_ok=True)
            (directory / 'stray.txt').write_text('junk', encoding='utf8')
            with self.assertRaises(ValueError):
                mt.run_one(out, c, device='cpu')


if __name__ == '__main__':
    unittest.main()
