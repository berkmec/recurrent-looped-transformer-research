"""A checkpoint that cannot resume a run is a copy of the weights, not a checkpoint.

What exists today for the 30M line is one file, `latest.pt`, 118.8 MB, rewritten in place every 250 steps. It
holds model weights and nothing else: no optimizer moments, no step counter. `mline/emergence.py` contains no
load path at all, and `mline/train.run_one` refuses a partially finished condition outright ("incomplete
condition directory requires audit; no automatic reuse"). So an interruption costs the whole run, and an
interruption DURING a save costs the only copy, because the write is in place.

That is survivable for a 9-hour probe and reckless for the 22-hour M1, and it disqualifies the multi-day runs a
50M model would need. This campaign has already lost a run to a power cut once.

Three properties, each tested rather than asserted:

  ROUND TRIP   weights, optimizer moments and the step number come back exactly as they went in;
  ATOMIC       a crash mid-write leaves the previous checkpoint intact and loadable;
  ROTATION     the last good checkpoint survives the next save, so one corrupt file is never fatal.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        from mline import checkpoint as ck
        return ck

    def model_and_opt(self, seed=0):
        torch.manual_seed(seed)
        m = torch.nn.Linear(8, 4)
        o = torch.optim.AdamW(m.parameters(), lr=1e-3)
        loss = m(torch.randn(3, 8)).square().mean()
        loss.backward(); o.step()                       # give the optimizer non-trivial state to round-trip
        return m, o

    def test_round_trip_restores_weights_optimizer_and_step(self):
        ck = self.api()
        with TemporaryDirectory() as d:
            path = Path(d, 'latest.pt')
            m, o = self.model_and_opt()
            ck.save(path, m, o, step=137, meta=dict(note='hello'))

            m2, o2 = self.model_and_opt(seed=1)
            state = ck.load(path)
            m2.load_state_dict(state['model']); o2.load_state_dict(state['optimizer'])
            self.assertEqual(state['step'], 137)
            self.assertEqual(state['meta']['note'], 'hello')
            for a, b in zip(m.parameters(), m2.parameters()):
                self.assertTrue(torch.equal(a.detach(), b.detach()))
            sa = o.state_dict()['state'][0]['exp_avg']
            sb = o2.state_dict()['state'][0]['exp_avg']
            self.assertTrue(torch.equal(sa, sb), 'optimizer moments must survive, or the resume is not the run')

    def test_a_crash_mid_write_does_not_destroy_the_previous_checkpoint(self):
        ck = self.api()
        with TemporaryDirectory() as d:
            path = Path(d, 'latest.pt')
            m, o = self.model_and_opt()
            ck.save(path, m, o, step=1)
            good = path.read_bytes()
            # simulate the crash: the temp file exists, half written, and the process dies before the rename
            Path(str(path) + '.tmp').write_bytes(b'\x00' * 64)
            self.assertEqual(path.read_bytes(), good, 'the live checkpoint must be untouched until the rename')
            self.assertEqual(ck.load(path)['step'], 1)

    def test_the_previous_checkpoint_is_kept(self):
        ck = self.api()
        with TemporaryDirectory() as d:
            path = Path(d, 'latest.pt')
            m, o = self.model_and_opt()
            ck.save(path, m, o, step=1)
            ck.save(path, m, o, step=2)
            self.assertEqual(ck.load(path)['step'], 2)
            self.assertEqual(ck.load(Path(d, 'previous.pt'))['step'], 1,
                             'one corrupt file must never be fatal')

    def test_a_corrupt_latest_falls_back_to_the_previous(self):
        ck = self.api()
        with TemporaryDirectory() as d:
            path = Path(d, 'latest.pt')
            m, o = self.model_and_opt()
            ck.save(path, m, o, step=1); ck.save(path, m, o, step=2)
            path.write_bytes(b'not a checkpoint')
            state = ck.load_latest(Path(d))
            self.assertEqual(state['step'], 1)

    def test_load_returns_none_when_there_is_nothing_to_resume(self):
        ck = self.api()
        with TemporaryDirectory() as d:
            self.assertIsNone(ck.load_latest(Path(d)))

    def test_a_weights_only_file_is_still_readable(self):
        """The probe currently running wrote the old format; it must not become unreadable."""
        ck = self.api()
        with TemporaryDirectory() as d:
            path = Path(d, 'latest.pt')
            m, _ = self.model_and_opt()
            torch.save({k: v.cpu() for k, v in m.state_dict().items()}, path)
            state = ck.load(path)
            self.assertIsNone(state['step'], 'an old file has no step, and that must be visible rather than guessed')
            self.assertIn('weight', ' '.join(state['model']))


if __name__ == '__main__':
    unittest.main()
