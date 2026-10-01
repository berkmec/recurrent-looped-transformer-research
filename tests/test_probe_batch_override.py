"""The launch probe must be able to measure a batch other than the frozen one, and report bytes per second.

The rented card has 24 GB against the 8 GB at home, which reopens two questions the local hardware settled by
force: how large a batch is worth using, and whether the three M1 arms should run CONCURRENTLY on one GPU rather
than one after another. The campaign's "GPU jobs serial" rule was adopted for a card that could not hold two of
them; it is a hardware fact, not a principle, and on a bigger card it must be re-decided by measurement.

Which way it goes is genuinely unknown. If the loop is compute-bound, three concurrent arms share one card and
gain nothing. If it is launch-bound, they overlap and M1 drops from 22 hours to about 8. I have already been
caught generalising a 5 % utilisation figure from the toy models to the 30M one, so this is measured on the
actual card before the plan is built on it.

Bytes per second is the comparable number across batch sizes; seconds per step is not, because a step at batch 64
moves twice the data of a step at batch 32.
"""
import inspect
import unittest


class ProbeSignatureTests(unittest.TestCase):
    def probe(self):
        from mline import train as mt
        return mt.probe

    def test_the_probe_accepts_batch_sizes(self):
        params = inspect.signature(self.probe()).parameters
        self.assertIn('batches', params, 'the probe must be able to sweep batch sizes on a larger card')

    def test_the_default_still_measures_the_frozen_batch(self):
        from mline import train as mt
        default = inspect.signature(self.probe()).parameters['batches'].default
        self.assertIsNone(default, 'omitting it must keep the old behaviour, the frozen protocol batch')
        self.assertEqual(mt.protocol()['batch'], 32)

    def test_the_cli_exposes_it(self):
        from pathlib import Path
        text = Path(__file__).resolve().parents[1].joinpath('mline/train.py').read_text(encoding='utf8')
        self.assertIn("'--batches'", text)

    def test_throughput_is_reported_not_just_step_time(self):
        from pathlib import Path
        text = Path(__file__).resolve().parents[1].joinpath('mline/train.py').read_text(encoding='utf8')
        self.assertIn('bytes/s', text, 'seconds per step is not comparable across batch sizes; bytes per second is')


if __name__ == '__main__':
    unittest.main()
