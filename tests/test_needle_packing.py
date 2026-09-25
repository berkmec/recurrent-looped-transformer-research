"""The needle training rows were 83.7 per cent padding, so the experiment's own signal got one gradient in a thousand.

Measured on the mix M0 actually trained on: a needle row padded to the 512-byte training length carried 83.2 bytes
of content and 428.8 bytes of leading spaces. With needle rows at 12.5 per cent of the batch that makes

    needle CONTENT            2.03 % of the training signal
    the four ANSWER bytes     0.098 %   - one part in a thousand
    "predict a space after a space"  10.47 %   - five times the needle content

and the M0 probe duly finished 81.9 MB with exact_match 0.000 everywhere, cross-entropy 4.070 nats on the answer
bytes of its OWN training distribution (uniform over 256 bytes is 5.545; knowing only that the answer is a digit
would be 2.303) and 26.6 per cent of its answer-position predictions not even digits. The model never learned the
format, because almost nothing asked it to.

The padding was mine: `pad_to` was added during the pre-flight fix so needle rows could be concatenated with
corpus rows, and the loss composition was never checked. Padding a row to length is not free - every padded
position is a training target, and here they outnumbered the real ones five to one.

The fix is to PACK: fill the row with as many needle examples as fit, so the content fraction is near one and no
position teaches the model to emit spaces. It changes no hypothesis and no threshold; it stops throwing away five
sixths of the budget the mix was given.
"""
import unittest

import torch


class PackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def nd(self):
        from mline import needle as nd
        return nd

    def rows(self, **kw):
        return self.nd().training_items(8, filler_range=(0, 64), seed=5, pad_to=512, **kw)

    def padding_fraction(self, x):
        lead = torch.tensor([float((row == 32).cumprod(0).sum()) for row in x])
        return float(lead.mean()) / x.shape[1]

    def test_the_flaw_is_real_when_packing_is_off(self):
        """Guards the premise, so this file cannot quietly become vacuous."""
        x = self.rows(pack=False)['x']
        self.assertGreater(self.padding_fraction(x), .7, 'the unpacked mix should be mostly padding')

    def test_packing_fills_the_row(self):
        x = self.rows(pack=True)['x']
        self.assertLess(self.padding_fraction(x), .1, 'a packed row must be almost entirely content')

    def test_packing_is_the_default(self):
        self.assertLess(self.padding_fraction(self.rows()['x']), .1)

    def test_a_packed_row_holds_several_complete_needles(self):
        items = self.rows(pack=True)
        for text in items['texts']:
            self.assertGreaterEqual(text.count('The code for'), 4,
                                    'each example appears twice - fact and question - so a filled row holds several')

    def test_targets_still_shift_by_one(self):
        items = self.rows(pack=True)
        self.assertTrue(torch.equal(items['x'][:, 1:], items['y'][:, :-1]))

    def test_keys_stay_inside_the_training_alphabet(self):
        nd = self.nd()
        allowed = set(nd.TRAIN_KEYS.decode('ascii'))
        for text in self.rows(pack=True)['texts']:
            for chunk in text.split('The code for ')[1:]:
                self.assertTrue(set(chunk[:4]) <= allowed, f'key {chunk[:4]!r} is outside TRAIN_KEYS')

    def test_the_answer_bytes_get_a_materially_larger_share(self):
        """The point of the change, stated as a number rather than as an intention."""
        from mline import train as mt
        p = mt.protocol()
        share = {}
        for pack in (False, True):
            items = self.nd().training_items(16, filler_range=tuple(p['needle_train_filler']), seed=3,
                                             pad_to=p['length'], pack=pack)
            answers = sum(4 * t.count('The code for') // 2 for t in items['texts']) / len(items['texts'])
            share[pack] = (p['needle_train_rate']) * answers / p['length']
        self.assertGreater(share[True], 5 * share[False],
                           f'packing must buy back most of the wasted budget: {share}')


if __name__ == '__main__':
    unittest.main()


class ObjectiveCompositionIsGuardedTests(unittest.TestCase):
    """The lesson from this failure, written as a computation rather than as a note.

    A pre-flight check had already confirmed the needle mix EXISTED in the runner - that check is why I felt safe
    to launch, and it measured the wrong property. Presence is not proportion. This one measures the share of the
    training objective that actually reaches the bytes the experiment reads, against a floor declared in the
    protocol, so a later change to the batch, the rate, the sequence length or the padding cannot quietly starve
    the signal again without a test going red.
    """

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def composition(self):
        """Per-component share of the next-byte objective in one M1 training batch, measured not assumed."""
        from mline import needle as nd
        from mline import train as mt
        p = mt.protocol()
        rows = round(p['batch'] * p['needle_train_rate'])
        mix = nd.training_items(64, filler_range=tuple(p['needle_train_filler']), seed=7, pad_to=p['length'])
        x = mix['x']
        pad = torch.tensor([float((r == 32).cumprod(0).sum()) for r in x]).mean().item()
        content = p['length'] - pad
        answers = 4 * (sum(mix['examples_per_row']) / len(mix['examples_per_row']))
        share = rows / p['batch'] / p['length']
        return dict(answer_bytes=share * answers, needle_content=share * content, padding=share * pad,
                    floor=p['preregistration']['resolution']['objective_floor'])

    def test_the_answer_bytes_clear_the_declared_floor(self):
        c = self.composition()
        self.assertGreater(c['answer_bytes'], c['floor'],
                           f'the bytes this experiment reads get {c["answer_bytes"]:.5f} of the objective, below '
                           f'the floor {c["floor"]}. That is how the first M0 probe learned nothing: 0.00098.')

    def test_padding_no_longer_outweighs_the_signal(self):
        c = self.composition()
        self.assertLess(c['padding'], c['needle_content'],
                        f'padding takes {c["padding"]:.4f} of the objective against {c["needle_content"]:.4f} for '
                        'real content; in the failed run it was 10.47 % against 2.03 %')

    def test_the_floor_is_registered_and_would_have_caught_the_failure(self):
        from mline import train as mt
        floor = mt.protocol()['preregistration']['resolution']['objective_floor']
        self.assertGreater(floor, .00098, 'the floor must exclude the composition that actually failed')
        self.assertLess(floor, self.composition()['answer_bytes'], 'a floor above the current mix is unfalsifiable')
