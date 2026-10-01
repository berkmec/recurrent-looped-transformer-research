"""U4-sep2 aggregation contract: the four pre-registered predictions, applied across pairs, as tested code."""
import unittest


class U4Sep2AnalyzeTests(unittest.TestCase):
    def api(self):
        try:
            from horizon import u4sep2_analyze as ua
        except ImportError as e:
            self.fail('horizon.u4sep2_analyze is not implemented: ' + str(e))
        return ua

    def pair(self, start, control, hinge, ratio_start=1.18, ratio_hinge=2.5, ratio_control=1.19,
             val_hinge=1.0, val_control=1.0):
        return dict(checkpoint='soft/s117', u2c_repair=start, control_repair=control, hinge_repair=hinge,
                    ratio_u2c=ratio_start, ratio_control=ratio_control, ratio_hinge=ratio_hinge,
                    validation_control=val_control, validation_hinge=val_hinge)

    def test_thresholds_are_the_pre_registered_ones(self):
        t = self.api().THRESHOLDS
        self.assertEqual(t['vs_start_pp'], 30.)     # "more than +30 pp over its own U2c starting value"
        self.assertEqual(t['vs_control_pp'], 20.)   # "and more than +20 pp over its paired control"
        self.assertEqual(t['validation_min'], .95)  # "validation stays >= 0.95 in both arms"
        self.assertEqual(t['control_drift_pp'], 10.)  # "the control arm's repair moves by less than +10 pp"
        self.assertEqual(t['ratio_target'], 1.5)

    def test_too_few_pairs_gives_no_verdict_at_all(self):
        """The machinery exists to prevent premature conclusions; with one pair, "a majority" is trivially true."""
        ua = self.api()
        v = ua.verdict([self.pair(-40.7, -40.7, 54.4)])
        self.assertEqual(v['pairs_rescued'], 1)
        self.assertEqual(v['status'], 'INCOMPLETE')
        self.assertEqual(ua.THRESHOLDS['min_pairs'], 3)

    def test_a_replicated_rescue_is_causal_support(self):
        ua = self.api()
        pairs = [self.pair(-40.7, -40.7, 54.4), self.pair(-30.7, -28.0, 41.0), self.pair(-99.5, -97.0, 10.0)]
        v = ua.verdict(pairs)
        self.assertEqual(v['pairs_rescued'], 3)     # each clears +30 vs start and +20 vs its control
        self.assertTrue(v['prediction_d_control_flat'])
        self.assertEqual(v['status'], 'CAUSAL_SUPPORTED')

    def test_a_single_success_among_failures_is_not_replication(self):
        ua = self.api()
        pairs = [self.pair(-40.7, -40.7, 54.4), self.pair(-30.7, -29.0, -28.0), self.pair(-99.5, -97.0, -95.0)]
        v = ua.verdict(pairs)
        self.assertEqual(v['pairs_rescued'], 1)
        self.assertEqual(v['status'], 'NOT_REPLICATED')

    def test_ratio_moved_but_repair_did_not_is_the_falsification(self):
        ua = self.api()
        pairs = [self.pair(-40.7, -40.7, -39.0), self.pair(-30.7, -29.0, -30.0), self.pair(-99.5, -97.0, -98.0)]
        v = ua.verdict(pairs)
        self.assertTrue(v['prediction_a_ratio_reached'])
        self.assertEqual(v['pairs_rescued'], 0)
        self.assertEqual(v['status'], 'CORRELATE_NOT_CAUSE')

    def test_a_control_that_also_improves_means_it_was_training(self):
        ua = self.api()
        pairs = [self.pair(-40.7, 20.0, 54.4), self.pair(-30.7, 15.0, 41.0), self.pair(-99.5, -40.0, 10.0)]
        v = ua.verdict(pairs)
        self.assertFalse(v['prediction_d_control_flat'])
        self.assertEqual(v['status'], 'CONFOUNDED_BY_TRAINING')

    def test_a_rescue_paid_for_in_accuracy_is_not_a_rescue(self):
        ua = self.api()
        pairs = [self.pair(-40.7, -40.7, 54.4, val_hinge=.61), self.pair(-30.7, -28.0, 41.0, val_hinge=.58),
                 self.pair(-99.5, -97.0, 10.0, val_hinge=.60)]
        v = ua.verdict(pairs)
        self.assertFalse(v['prediction_c_accuracy_kept'])
        self.assertEqual(v['status'], 'RESCUE_AT_A_COST')

    def test_a_hinge_that_never_reached_the_target_tested_nothing(self):
        ua = self.api()
        pairs = [self.pair(-40.7, -40.7, -40.0, ratio_hinge=1.2), self.pair(-30.7, -29.0, -30.0, ratio_hinge=1.1),
                 self.pair(-99.5, -97.0, -98.0, ratio_hinge=1.25)]
        v = ua.verdict(pairs)
        self.assertFalse(v['prediction_a_ratio_reached'])
        self.assertEqual(v['status'], 'INTERVENTION_INEFFECTIVE')


if __name__ == '__main__':
    unittest.main()
