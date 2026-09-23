"""M1 verdict contract, written before the 30M models exist."""
import unittest


class M1AnalyzeTests(unittest.TestCase):
    def api(self):
        try:
            from mline import analyze
        except ImportError as e:
            self.fail('mline.analyze is not implemented: ' + str(e))
        return analyze

    def arms(self, plain_bpb=1.60, hard_bpb=1.60, soft_bpb=1.62,
             plain_needle=(0.95, 0.30, 0.10), hard_needle=(0.95, 0.60, 0.45), soft_needle=(0.94, 0.35, 0.15)):
        def row(bpb, needle):
            return dict(bits_per_byte=bpb, needle={'repeat@0': needle[0], 'repeat@512': needle[1], 'repeat@2048': needle[2],
                                                   'text@0': needle[0], 'text@512': needle[1], 'text@2048': needle[2]})
        return dict(plain=row(plain_bpb, plain_needle), hard_st=row(hard_bpb, hard_needle), soft=row(soft_bpb, soft_needle))

    def test_thresholds_are_the_pre_registered_ones(self):
        t = self.api().THRESHOLDS
        self.assertEqual(t['bpb_regression'], .02)   # "does not hurt the language ability"
        self.assertEqual(t['needle_gain'], .05)      # "improves needle@512 and needle@2048"
        self.assertEqual(t['long_filler'], [512, 2048])

    def test_gate_that_helps_without_costing_language_transfers(self):
        ua = self.api()
        v = ua.verdict(self.arms())
        self.assertTrue(v['prediction_a_no_bpb_cost'])
        self.assertTrue(v['prediction_b_needle_gain'])
        self.assertTrue(v['prediction_c_soft_below_hard'])
        self.assertEqual(v['status'], 'GATE_TRANSFERS')

    def test_a_gate_that_buys_needles_with_perplexity_is_flagged(self):
        ua = self.api()
        v = ua.verdict(self.arms(hard_bpb=1.75))
        self.assertFalse(v['prediction_a_no_bpb_cost'])
        self.assertEqual(v['status'], 'GATE_AT_A_COST')

    def test_no_needle_gain_means_the_toy_result_did_not_transfer(self):
        ua = self.api()
        v = ua.verdict(self.arms(hard_needle=(0.95, 0.31, 0.11)))
        self.assertFalse(v['prediction_b_needle_gain'])
        self.assertEqual(v['status'], 'GATE_DOES_NOT_TRANSFER')

    def test_soft_beating_hard_contradicts_the_toy_prediction_and_is_recorded(self):
        ua = self.api()
        v = ua.verdict(self.arms(soft_needle=(0.95, 0.80, 0.70)))
        self.assertFalse(v['prediction_c_soft_below_hard'])
        self.assertIn('soft', v['notes'].lower())

    def test_a_missing_arm_gives_no_verdict(self):
        ua = self.api()
        arms = self.arms(); arms.pop('soft')
        self.assertEqual(ua.verdict(arms)['status'], 'INCOMPLETE')

    def test_single_seed_is_marked_provisional(self):
        ua = self.api()
        v = ua.verdict(self.arms(), seeds=1)
        self.assertTrue(v['provisional'])
        v2 = ua.verdict(self.arms(), seeds=3)
        self.assertFalse(v2['provisional'])


if __name__ == '__main__':
    unittest.main()
