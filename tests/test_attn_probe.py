"""Where does the answer position look?

The decoy probe says WHICH fact's value comes out; this says where the decoder's cross-attention is looking when it decides:
how much weight lands on the asked fact's value digits, on the other fact's, on the keys. A model that answers from the right
fact because it matched the key shows key-conditional attention; a position-locked one looks at the same place whichever key
is asked. The capture must therefore be exact (the Core's output unchanged, the weights a distribution over the memory it could
see) and the regions must tile the row.
"""
import unittest

import torch

from core.model import Core
from mline import attn_probe as ap
from mline import decoy_probe as dp
from mline import needle as nd


def small_core(feedback, seed=0):
    torch.manual_seed(seed)
    core = Core(width=16, heads=2, window=4, classes=256, feedback=feedback)
    core.embedding = torch.nn.Embedding(256, 16)
    return core.eval()


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.randint(0, 256, (3, 24))

    def test_the_weights_are_a_distribution_over_the_memory_visible_at_that_position(self):
        for feedback in (True, False):
            with self.subTest(feedback=feedback):
                weights = ap.cross_attention_at(small_core(feedback), self.x, position=9)
                self.assertEqual(tuple(weights.shape), (3, 2, 10))              # batch, heads, positions 0..9
                self.assertTrue((weights >= 0).all())
                self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(3, 2), atol=1e-5))

    def test_capturing_does_not_change_the_models_output(self):
        for feedback in (True, False):
            with self.subTest(feedback=feedback):
                core = small_core(feedback)
                with torch.no_grad():
                    plain = core(self.x)
                    ap.cross_attention_at(core, self.x, position=5)
                    again = core(self.x)
                self.assertTrue(torch.equal(plain, again))

    def test_the_hook_is_removed_afterwards_even_when_the_forward_fails(self):
        core = small_core(True)
        ap.cross_attention_at(core, self.x, position=3)
        self.assertNotIn('attend', core.__dict__)
        with self.assertRaises(ValueError):
            ap.cross_attention_at(core, self.x, position=99)
        self.assertNotIn('attend', core.__dict__)

    def test_the_sequential_and_parallel_paths_agree_when_there_is_no_feedback_to_differ_on(self):
        """With the feedback switched off, forward() takes the parallel path; stepping the decoder one position at a time
        (the sequential path) over the same weights must show the same cross-attention."""
        core = small_core(False)
        parallel = ap.cross_attention_at(core, self.x, position=7)
        sequential = ap.cross_attention_at(core, self.x, position=7, force_sequential=True)
        self.assertTrue(torch.allclose(parallel, sequential, atol=1e-5))

    def test_a_wrapped_model_is_refused(self):
        class Wrapper(torch.nn.Module):
            def forward(self, x):
                return x
        with self.assertRaises(TypeError):
            ap.cross_attention_at(Wrapper(), self.x, position=1)


class RegionTests(unittest.TestCase):
    def layout(self, key_length=4, a=5, b=7):
        return ap.regions(key_length, a, b)

    def test_the_regions_tile_the_row_exactly(self):
        for key_length, a, b in ((4, 5, 7), (1, 0, 0), (2, 30, 0), (3, 0, 30)):
            with self.subTest(key_length=key_length, a=a, b=b):
                sets = dp.make(2, a, b, 'first', 1, nd.TRAIN_KEYS, key_length)
                spans = sorted(self.layout(key_length, a, b).values())
                self.assertEqual(spans[0][0], 0)
                for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
                    self.assertEqual(e1, s2)
                self.assertEqual(spans[-1][1], sets['width'])

    def test_each_region_holds_what_it_is_named_for(self):
        sets = dp.make(4, 5, 7, 'decoy', 2, nd.TRAIN_KEYS, 4)
        spans = self.layout(4, 5, 7)
        for i in range(4):
            row = bytes(sets['x'][i].tolist()).decode('ascii')
            cut = lambda name: row[spans[name][0]:spans[name][1]]
            self.assertEqual(cut('key_first'), sets['keys'][i])
            self.assertEqual(cut('value_first'), sets['answers'][i])
            self.assertEqual(cut('key_decoy'), sets['decoy_keys'][i])
            self.assertEqual(cut('value_decoy'), sets['decoy_answers'][i])
            self.assertEqual(cut('key_question'), sets['decoy_keys'][i])      # ask == 'decoy'
            self.assertEqual(cut('answer'), sets['decoy_answers'][i])
            self.assertEqual(len(cut('filler_a')), 5)
            self.assertEqual(len(cut('filler_b')), 7)

    def test_the_answer_position_is_the_last_visible_byte_before_the_answer(self):
        sets = dp.make(2, 5, 7, 'first', 3, nd.TRAIN_KEYS, 4)
        self.assertEqual(ap.regions(4, 5, 7)['answer'][0], sets['first'] + 1)

    def test_masses_sum_to_one_over_the_visible_regions(self):
        weights = torch.softmax(torch.randn(4, 2, 70), -1)                      # position 69 sees 70 bytes
        spans = ap.regions(4, 5, 7)
        masses = ap.region_masses(weights, spans, position=69)
        self.assertAlmostEqual(sum(masses[name].sum().item() for name in masses) / (4 * 2), 1.0, places=5)
        self.assertEqual(tuple(masses['key_first'].shape), (4, 2))
        self.assertNotIn('answer', masses)                                      # the answer digits are not visible yet


class SelectivityTests(unittest.TestCase):
    def test_selectivity_is_the_share_of_the_two_value_regions_that_lands_on_the_asked_fact(self):
        masses = dict(value_first=torch.tensor([[.6, .1]]), value_decoy=torch.tensor([[.2, .3]]))
        self.assertTrue(torch.allclose(ap.selectivity(masses, 'first'), torch.tensor([[.75, .25]])))
        self.assertTrue(torch.allclose(ap.selectivity(masses, 'decoy'), torch.tensor([[.25, .75]])))

    def test_a_head_that_looks_at_neither_value_is_undefined_not_zero(self):
        masses = dict(value_first=torch.zeros(1, 1), value_decoy=torch.zeros(1, 1))
        self.assertTrue(torch.isnan(ap.selectivity(masses, 'first')).all())


class RunTests(unittest.TestCase):
    def test_a_run_reports_masses_and_selectivity_per_head_for_both_asks(self):
        report = ap.probe(small_core(True), [(0, 0), (5, 3)], asks=('first', 'decoy'), alphabet='TRAIN_KEYS', key_length=2,
                          items=4, batch=2, device='cpu')
        self.assertEqual(len(report['rows']), 2 * 2)
        row = report['rows'][0]
        for field in ('a', 'b', 'ask', 'masses', 'selectivity', 'heads'):
            self.assertIn(field, row)
        self.assertEqual(row['heads'], 2)
        self.assertEqual(len(row['selectivity']), 2)                             # one number per head
        self.assertAlmostEqual(sum(sum(v) for v in row['masses'].values()), 2.0, places=4)   # per head the masses sum to 1; two heads


if __name__ == '__main__':
    unittest.main()
