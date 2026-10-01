"""M1's two decisive needle points are LONGER than anything the model is trained on, so a failure there is ambiguous.

Measured from the code, before any 30M arm exists: training length is 512 bytes, while the evaluation sequences are
565 bytes at filler 512 (1.10x) and 2101 bytes at filler 2048 (4.10x). Prediction (b) - the one whose failure stops
the M-line - demands a needle gain at exactly those two points. As it stands, an arm that scores zero there could be
failing for either of two completely different reasons:

  (i)  it cannot carry information across the distance          <- what the experiment is about
  (ii) it cannot process a sequence 4x longer than it was       <- a property of rotary/positional extrapolation,
       trained on, at any distance                                 nothing to do with the gate

A length-matched control separates them. `placement='near'` builds a sequence of the IDENTICAL width in which the
filler comes first and the fact sits immediately before the question, so the distance is ~0 while the length is
unchanged. An arm that fails `far` but passes `near` at the same width has a memory failure. An arm that fails both
has a length failure, and its `far` score is not evidence about the gate either way.

This is the same move the campaign made with U2b's freeze controls: before believing a drop means what you want it
to mean, build the condition that would show the drop has a duller cause.
"""
import unittest

import torch


class LengthMatchedControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def nd(self):
        from mline import needle as nd
        return nd

    def test_near_and_far_have_the_identical_width(self):
        nd = self.nd()
        for n in (128, 512, 2048):
            far = nd.make(6, n_filler=n, kind='repeat', seed=5, placement='far')
            near = nd.make(6, n_filler=n, kind='repeat', seed=5, placement='near')
            self.assertEqual(far['width'], near['width'],
                             f'the control only isolates distance if the LENGTH is held fixed (filler {n})')
            self.assertEqual(far['x'].shape, near['x'].shape)

    def test_near_puts_the_fact_next_to_the_question_and_far_does_not(self):
        nd = self.nd()
        n = 512
        for placement, want in (('far', n), ('near', 0)):
            items = nd.make(4, n_filler=n, kind='repeat', seed=7, placement=placement)
            self.assertEqual(items['placement'], placement)
            for d in items['distances']:
                self.assertEqual(d, want, f'{placement}: fact-to-question distance should be {want}, got {d}')

    def test_the_answer_is_still_the_last_thing_in_the_sequence(self):
        nd = self.nd()
        for placement in ('far', 'near'):
            items = nd.make(4, n_filler=256, kind='repeat', seed=9, placement=placement)
            mask, width = items['mask'], items['width']
            answer_len = len(items['answers'][0])
            for row in range(mask.shape[0]):
                on = mask[row].nonzero().flatten().tolist()
                self.assertEqual(on, list(range(width - answer_len - 1, width - 1)),
                                 f'{placement}: the scored span must remain the final answer bytes')

    def test_near_still_contains_the_fact_and_the_question(self):
        nd = self.nd()
        items = nd.make(3, n_filler=300, kind='repeat', seed=11, placement='near')
        for row, key, value in zip(items['x'], items['answers_keys'], items['answers']):
            text = bytes(row.tolist()).decode('ascii', errors='replace')
            self.assertEqual(text.count(f'The code for {key} is'), 2, 'fact and question must both be present')
            self.assertTrue(text.rstrip().endswith(value))

    def test_far_is_unchanged_from_the_pre_control_behaviour(self):
        """The control must not silently alter the condition it is a control FOR."""
        nd = self.nd()
        a = nd.make(5, n_filler=512, kind='repeat', seed=3)                      # default
        b = nd.make(5, n_filler=512, kind='repeat', seed=3, placement='far')
        self.assertTrue(torch.equal(a['x'], b['x']))
        self.assertTrue(torch.equal(a['y'], b['y']))
        self.assertTrue(torch.equal(a['mask'], b['mask']))
        self.assertEqual(a['width'], 565)                                        # the measured width, pinned

    def test_text_filler_also_supports_the_control(self):
        nd = self.nd()
        import numpy as np
        corpus = np.frombuffer(bytes(range(32, 127)) * 200, dtype=np.uint8)   # md.load returns a uint8 array
        far = nd.make(3, n_filler=400, kind='text', seed=4, corpus=corpus, placement='far')
        near = nd.make(3, n_filler=400, kind='text', seed=4, corpus=corpus, placement='near')
        self.assertEqual(far['width'], near['width'])


class ProtocolAndVerdictUseTheControlTests(unittest.TestCase):
    def test_protocol_evaluates_both_placements_at_the_over_length_fillers(self):
        from mline import train as mt
        p = mt.protocol()
        over = [n for n in p['needle_filler'] if n > 0]
        self.assertTrue(set(p['needle_placements']) == {'far', 'near'},
                        'both placements must be evaluated or the ambiguity stays unresolved')
        for n in p['long_filler'] if 'long_filler' in p else [512, 2048]:
            self.assertIn(n, over)

    def test_verdict_reports_length_failure_rather_than_gate_failure(self):
        """An arm that cannot do the task at distance ~0 and that length has not been measured on memory."""
        from mline import analyze as ma

        def arm(far, near):
            return dict(bits_per_byte=1.5,
                        needle={f'repeat@{n}': far for n in (512, 2048)}
                               | {f'text@{n}': far for n in (512, 2048)}
                               | {f'repeat@{n}@near': near for n in (512, 2048)}
                               | {f'text@{n}@near': near for n in (512, 2048)})

        # every arm is at the floor BOTH far and near: the sequence length defeated them, not the distance
        blind = dict(plain=arm(0., 0.), hard_st=arm(0., 0.), soft=arm(0., 0.))
        self.assertEqual(ma.verdict(blind)['status'], 'NOT_MEASURABLE_AT_THIS_LENGTH')

        # the arms CAN do it next to the question but not across the filler: a real memory result
        real = dict(plain=arm(0., .9), hard_st=arm(0., .9), soft=arm(0., .9))
        self.assertEqual(ma.verdict(real)['status'], 'GATE_DOES_NOT_TRANSFER')

    def test_length_branch_takes_priority_over_the_budget_floor(self):
        """Both say 'not measurable'; they must not be confusable, because the remedies differ."""
        from mline import analyze as ma
        v = ma.verdict(dict(
            plain=dict(bits_per_byte=1.5, needle={'repeat@512': 0., 'repeat@2048': 0., 'text@512': 0., 'text@2048': 0.,
                                                  'repeat@512@near': 0., 'repeat@2048@near': 0.,
                                                  'text@512@near': 0., 'text@2048@near': 0.}),
            hard_st=dict(bits_per_byte=1.5, needle={'repeat@512': 0., 'repeat@2048': 0., 'text@512': 0., 'text@2048': 0.,
                                                    'repeat@512@near': 0., 'repeat@2048@near': 0.,
                                                    'text@512@near': 0., 'text@2048@near': 0.}),
            soft=dict(bits_per_byte=1.5, needle={'repeat@512': 0., 'repeat@2048': 0., 'text@512': 0., 'text@2048': 0.,
                                                 'repeat@512@near': 0., 'repeat@2048@near': 0.,
                                                 'text@512@near': 0., 'text@2048@near': 0.})))
        self.assertEqual(v['status'], 'NOT_MEASURABLE_AT_THIS_LENGTH')
        self.assertIn('length', v['meaning'].lower())


if __name__ == '__main__':
    unittest.main()
