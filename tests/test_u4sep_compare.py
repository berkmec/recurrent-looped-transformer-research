"""U4-sep verdict contract, written BEFORE the sweep produced any number.

The point of this file is that the decision rule between "the U2d finding is causal" and "it was a correlate" is
fixed in code, with tests, while the results are still unknown. Anything the runs produce is then scored by a rule
that could not have been shaped by the results.
"""
import unittest


class RepairFractionTests(unittest.TestCase):
    def api(self):
        try:
            from horizon import u4sep_compare as uc
        except ImportError as e:
            self.fail('horizon.u4sep_compare is not implemented: ' + str(e))
        return uc

    def test_repair_fraction_matches_the_u2c_definition(self):
        """(arm - none) / (oracle - none), in per cent: the U2c ledger's 'fraction of the oracle's available repair'."""
        uc = self.api()
        # U2c soft/five/s117 at lr 0.003: none 0.430, oracle 1.000, arm 1.000 -> +100
        self.assertAlmostEqual(uc.repair_fraction(1.0, 0.430, 1.0), 100.0, places=3)
        # the same condition at lr 0.0003: arm 0.198 -> about -41
        self.assertAlmostEqual(uc.repair_fraction(0.198, 0.430, 1.0), -40.70175, places=3)
        self.assertAlmostEqual(uc.repair_fraction(0.430, 0.430, 1.0), 0.0, places=6)   # no change = no repair
        self.assertIsNone(uc.repair_fraction(0.5, 0.9, 0.9))                            # no headroom -> undefined

    def test_verdict_confirms_only_when_robustness_follows_the_ratio(self):
        uc = self.api()
        causal = uc.verdict(low=dict(repair=[55., 61., 48.], validation=[1., 1., .99], sep_ratio=[8.1, 7.4, 9.0]),
                            high=dict(repair=[100., 100., 98.], validation=[1., 1., 1.], sep_ratio=[30., 28., 33.]),
                            baseline=dict(low_repair=[-40.7, -41., -47.], low_sep=[3.4, 2.9, .3], high_repair=[100., 100., 100.], high_sep=[21., 33., 13.]))
        self.assertTrue(causal['prediction_a_low_lr_rescued'])
        self.assertTrue(causal['prediction_c_ratio_rose'])
        self.assertEqual(causal['status'], 'CAUSAL_SUPPORTED')

    def test_verdict_reports_a_mere_correlate_when_the_ratio_rises_alone(self):
        uc = self.api()
        v = uc.verdict(low=dict(repair=[-38., -44., -40.], validation=[1., 1., 1.], sep_ratio=[9.5, 8.8, 10.2]),
                       high=dict(repair=[100., 99., 100.], validation=[1., 1., 1.], sep_ratio=[31., 29., 34.]),
                       baseline=dict(low_repair=[-40.7, -41., -47.], low_sep=[3.4, 2.9, .3], high_repair=[100., 100., 100.], high_sep=[21., 33., 13.]))
        self.assertTrue(v['prediction_c_ratio_rose'])
        self.assertFalse(v['prediction_a_low_lr_rescued'])
        self.assertEqual(v['status'], 'CORRELATE_NOT_CAUSE')

    def test_verdict_flags_a_rescue_that_was_paid_for_in_accuracy(self):
        """Repair that arrives with a collapsed validation accuracy is not a rescue, it is a different failure."""
        uc = self.api()
        v = uc.verdict(low=dict(repair=[70., 65., 72.], validation=[.61, .58, .60], sep_ratio=[9., 8., 10.]),
                       high=dict(repair=[100., 100., 100.], validation=[1., 1., 1.], sep_ratio=[30., 30., 30.]),
                       baseline=dict(low_repair=[-40.7, -41., -47.], low_sep=[3.4, 2.9, .3], high_repair=[100., 100., 100.], high_sep=[21., 33., 13.]))
        self.assertFalse(v['prediction_a_low_lr_rescued'])
        self.assertEqual(v['status'], 'RESCUE_AT_A_COST')

    def test_verdict_notices_when_the_loss_did_not_move_the_ratio_at_all(self):
        """If the auxiliary loss failed to widen the ratio, the experiment tested nothing and must say so."""
        uc = self.api()
        v = uc.verdict(low=dict(repair=[-40., -42., -45.], validation=[1., 1., 1.], sep_ratio=[3.5, 3.0, .4]),
                       high=dict(repair=[100., 100., 100.], validation=[1., 1., 1.], sep_ratio=[21., 33., 13.]),
                       baseline=dict(low_repair=[-40.7, -41., -47.], low_sep=[3.4, 2.9, .3], high_repair=[100., 100., 100.], high_sep=[21., 33., 13.]))
        self.assertFalse(v['prediction_c_ratio_rose'])
        self.assertEqual(v['status'], 'INTERVENTION_INEFFECTIVE')

    def test_verdict_separates_a_rescue_that_damaged_the_arm_that_already_worked(self):
        uc = self.api()
        v = uc.verdict(low=dict(repair=[55., 61., 48.], validation=[1., 1., .99], sep_ratio=[8.1, 7.4, 9.0]),
                       high=dict(repair=[40., 35., 20.], validation=[1., 1., 1.], sep_ratio=[30., 28., 33.]),
                       baseline=dict(low_repair=[-40.7, -41., -47.], low_sep=[3.4, 2.9, .3], high_repair=[100., 100., 100.], high_sep=[21., 33., 13.]))
        self.assertTrue(v['prediction_a_low_lr_rescued'])
        self.assertFalse(v['prediction_b_high_lr_unchanged'])
        self.assertEqual(v['status'], 'CAUSAL_SUPPORTED_WITH_HIGH_LR_COST')

    def test_thresholds_are_the_pre_registered_ones(self):
        uc = self.api()
        t = uc.THRESHOLDS
        self.assertEqual(t['repair_pp'], 30.)          # "clearly positive (> +30 %)"
        self.assertEqual(t['validation_min'], .95)     # "while validation stays ~1.000"
        self.assertEqual(t['high_lr_drop_pp'], 10.)    # "should change little (no room)"
        self.assertEqual(t['ratio_rise'], 1.25)        # a rise worth calling a rise: +25 % on the mean


if __name__ == '__main__':
    unittest.main()
