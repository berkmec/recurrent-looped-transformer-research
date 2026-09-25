"""M1's pre-registered effect size was smaller than the noise of the measurement it is read from.

This is arithmetic, not a result: it needs no 30M model and no probe output, only the evaluation's item count.

`needle_gain = 0.05` asks hard_st to beat plain by five points of exact match at both long fillers. The needle sets
held 64 items, so 0.05 is 3.2 items. The standard error of a difference of two proportions at n=64 per arm is
sqrt(2p(1-p)/64), which is 0.053 at p = 0.10 and 0.081 at p = 0.30. For any score above about seven per cent the
threshold therefore sat BELOW one standard error of its own sampling noise, and a gain of exactly 0.05 would have
been a coin flip reported as GATE_TRANSFERS.

The thresholds were chosen while the needle task was still the zero-shot version - the one whose format no arm was
ever taught, which produced five flat checkpoints below chance. After the task was redesigned (format taught at
short filler, distance generalised) nothing rechecked whether the numbers still described a measurement that could
resolve them. That is the whole of the defect: an inherited constant.

Two changes, both made before any M1 arm exists:

  - the evaluation gets resolution rather than the rule getting blunter: 256 items instead of 64. It is also
    CHEAPER than the old setting, because `needle_batch` rises from 4 to 32 at the same time - eight batches of 32
    against sixteen of 4 is half the launches, and this workload is launch-bound;
  - the rule stops trusting a fixed constant: a gain must clear the pre-registered 0.05 AND two standard errors
    computed from the scores actually observed. Where both arms sit near zero the constant binds; where they sit
    mid-range the noise term binds. It can only make a positive claim harder, never easier.
"""
import math
import unittest


def se_of_difference(p, n):
    return math.sqrt(2 * p * (1 - p) / n)


class ResolutionTests(unittest.TestCase):
    def protocol(self):
        from mline import train as mt
        return mt.protocol()

    def thresholds(self):
        from mline import analyze as ma
        return ma.THRESHOLDS

    def test_the_item_count_can_resolve_the_pre_registered_effect(self):
        n = self.protocol()['needle_items']
        gain = self.thresholds()['needle_gain']
        worst = se_of_difference(.3, n)                     # mid-range scores are the noisiest case that matters
        self.assertLess(worst, gain,
                        f'at {n} items the sampling noise is {worst:.3f}, which swamps a {gain} threshold')

    def test_the_floor_is_at_least_one_item(self):
        """A floor below 1/n is unreachable from below: no arm can score between 0 and it."""
        p, t = self.protocol(), self.thresholds()
        self.assertGreaterEqual(t['measurable_floor'], 1 / p['needle_items'],
                                'a floor finer than the measurement grid cannot be tested')

    def test_the_evaluation_did_not_get_more_expensive(self):
        """Resolution was bought with batching, not with GPU hours, on a launch-bound workload."""
        p = self.protocol()
        launches = math.ceil(p['needle_items'] / p['needle_batch'])
        self.assertLessEqual(launches, math.ceil(64 / 4), 'more launches than the 64-item, batch-4 setting it replaces')

    def test_the_rule_requires_the_gain_to_clear_the_noise_as_well(self):
        from mline import analyze as ma

        def arm(bpb, far, near=.9):
            return dict(bits_per_byte=bpb,
                        needle={f'{k}@{n}': far for k in ('repeat', 'text') for n in (512, 2048)}
                               | {f'{k}@{n}@near': near for k in ('repeat', 'text') for n in (512, 2048)})

        # a gain of 0.06 clears the 0.05 constant but not two standard errors at these mid-range scores
        noisy = dict(plain=arm(1.5, .30), hard_st=arm(1.5, .36), soft=arm(1.5, .28))
        self.assertFalse(ma.verdict(noisy, items=256)['prediction_b_needle_gain'],
                         'a gain inside its own sampling noise must not count')
        self.assertEqual(ma.verdict(noisy, items=256)['status'], 'GATE_DOES_NOT_TRANSFER')

        # the same 0.06 gain near the floor IS outside the noise, because the noise shrinks with p
        small = dict(plain=arm(1.5, .01), hard_st=arm(1.5, .07), soft=arm(1.5, .01))
        self.assertTrue(ma.verdict(small, items=256)['prediction_b_needle_gain'])

    def test_the_noise_term_is_recorded_in_the_verdict(self):
        from mline import analyze as ma

        def arm(far):
            return dict(bits_per_byte=1.5,
                        needle={f'{k}@{n}': far for k in ('repeat', 'text') for n in (512, 2048)}
                               | {f'{k}@{n}@near': .9 for k in ('repeat', 'text') for n in (512, 2048)})

        v = ma.verdict(dict(plain=arm(.30), hard_st=arm(.36), soft=arm(.28)), items=256)
        self.assertIn('required_gain', v)
        for n in (512, 2048):
            self.assertGreater(v['required_gain'][n], ma.THRESHOLDS['needle_gain'],
                               'at mid-range scores the noise term must be the binding one')

    def test_the_resolution_argument_is_registered_with_the_protocol(self):
        pre = self.protocol()['preregistration']
        self.assertIn('resolution', pre)
        self.assertEqual(pre['resolution']['items'], self.protocol()['needle_items'])
        self.assertIn('standard error', pre['resolution']['basis'])


if __name__ == '__main__':
    unittest.main()
