"""The padded mix must remain runnable, because it is the CONTROL arm of the packing comparison.

Packing was adopted on an argument about the objective's composition - the four answer bytes went from 0.098 % of
the gradient to 0.543 % - and that argument needs no experiment. What still needs one is whether the change
produces the capability. The comparison is: same seed, same 5000 steps, same everything, one term different.

That term is `pack`. With it fixed at True the control cannot be run at all, and a comparison whose control is
unbuildable is not a comparison. The first probe's padded numbers exist, but they were measured on a different
machine, GPU, operating system and torch version, so using them as the control would put four extra differences
into a design that is supposed to carry one.
"""
import inspect
import unittest

import torch


class ProbeExposesThePackingTermTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        from mline import emergence as em
        return em

    def test_run_takes_the_packing_term(self):
        self.assertIn('pack', inspect.signature(self.api().run).parameters)

    def test_packing_is_the_default(self):
        self.assertIs(inspect.signature(self.api().run).parameters['pack'].default, True)

    def test_the_cli_can_select_either_arm(self):
        from pathlib import Path
        text = Path(__file__).resolve().parents[1].joinpath('mline/emergence.py').read_text(encoding='utf8')
        self.assertIn("'--no-pack'", text)

    def test_the_header_records_which_arm_ran(self):
        """A run whose record does not say which arm it is cannot be paired with anything later."""
        from pathlib import Path
        text = Path(__file__).resolve().parents[1].joinpath('mline/emergence.py').read_text(encoding='utf8')
        self.assertIn('pack=pack', text)

    def test_the_arms_share_their_first_draw_and_then_diverge_by_design(self):
        """The shapes match and the first needle is identical; beyond that the arms MUST diverge.

        Packing consumes more random draws per row, so from the second row onwards the two arms hold different
        needle examples. That is not a confound leaking in - it IS the treatment. The manipulation is 'more real
        needle content per row instead of padding', so the packed arm necessarily sees more examples. What must
        stay identical is everything else: the shape, the filler range, and the corpus rows, which come from
        md.batch(train, seed, step, ...) and never touch this generator at all.
        """
        from mline import needle as nd
        packed = nd.training_items(4, filler_range=(0, 64), seed=11, pad_to=512, pack=True)
        padded = nd.training_items(4, filler_range=(0, 64), seed=11, pad_to=512, pack=False)
        self.assertEqual(packed['x'].shape, padded['x'].shape)
        self.assertEqual(packed['filler_range'], padded['filler_range'])
        self.assertTrue(packed['texts'][0].startswith(padded['texts'][0][:40]),
                        'the first draw must be shared, or the seeds are not doing the same thing')
        self.assertGreater(sum(packed['examples_per_row']), sum(padded['examples_per_row']),
                           'the packed arm must carry more needle content; that is the whole intervention')
        self.assertTrue(packed['packed'])
        self.assertFalse(padded['packed'])


if __name__ == '__main__':
    unittest.main()
