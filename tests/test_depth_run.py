"""Z1 arms and runner mechanics (not learning outcomes)."""
import tempfile
import unittest
from pathlib import Path

import torch

from horizon import depth_run as dr
from horizon import depth_tasks as dt
from horizon.depth_models import ARMS, make_arm, parameter_count


class Arms(unittest.TestCase):
    def test_shapes_for_every_arm_and_task(self):
        for task in ('perm3', 'count'):
            x, _ = dt.generate(task, 3, 10, seed=1)
            for arm in ARMS:
                m = make_arm(arm, task, 32, seed=1)
                self.assertEqual(tuple(m(x).shape), (3, 10, dt.classes(task)), (task, arm))

    def test_core_embedding_matches_the_task_vocabulary(self):
        m = make_arm('core', 'perm5', 32, seed=1)
        self.assertEqual(m.embedding.num_embeddings, 120)
        self.assertTrue(m.feedback)
        self.assertFalse(make_arm('core_nofb', 'perm5', 32, seed=1).feedback)

    def test_same_seed_same_weights_different_seed_different(self):
        a = make_arm('core', 'perm3', 32, seed=4)
        b = make_arm('core', 'perm3', 32, seed=4)
        c = make_arm('core', 'perm3', 32, seed=5)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())))
        self.assertFalse(all(torch.equal(p, q) for p, q in zip(a.parameters(), c.parameters())))
        self.assertGreater(parameter_count(a), 10000)

    def test_unknown_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            make_arm('mamba', 'perm3', 32, seed=1)

    def test_every_arm_is_causal(self):
        # a model that peeks at later tokens would "solve" every prefix task for the wrong reason
        torch.manual_seed(0)
        x, _ = dt.generate('perm3', 2, 12, seed=3)
        x2 = x.clone()
        x2[:, 7:] = (x2[:, 7:] + 1) % dt.vocab('perm3')            # change everything from position 7 on
        for arm in ARMS:
            m = make_arm(arm, 'perm3', 32, seed=2)
            m.eval()
            with torch.no_grad():
                a, b = m(x), m(x2)
            self.assertTrue(torch.allclose(a[:, :7], b[:, :7], atol=1e-5), arm)
            self.assertFalse(torch.allclose(a[:, 7:], b[:, 7:], atol=1e-5), arm)       # and the later positions do react

    def test_lstm_and_transformer_baselines_exist_with_sensible_sizes(self):
        self.assertIn('lstm', ARMS)
        self.assertIn('tfm', ARMS)
        lstm = make_arm('lstm', 'perm3', 32, seed=1)
        tfm = make_arm('tfm', 'perm3', 64, seed=1)
        self.assertTrue(20_000 < parameter_count(make_arm('lstm', 'perm3', 64, seed=1)) < 60_000)
        self.assertTrue(50_000 < parameter_count(tfm) < 300_000, parameter_count(tfm))
        self.assertEqual(parameter_count(lstm) > parameter_count(make_arm('gru', 'perm3', 32, seed=1)), True)    # four gates against three

    def test_the_transformer_baseline_has_no_position_signal_of_its_own(self):
        # NoPE: permuting the prefix tokens leaves the representation of an identical-token prefix position-independent, so a sequence of one
        # repeated token gives the same output at every position (only causal attention can tell positions apart, and it cannot here)
        tfm = make_arm('tfm', 'perm3', 32, seed=1)
        tfm.eval()
        x = torch.zeros(1, 9, dtype=torch.long)
        with torch.no_grad():
            out = tfm(x)
        self.assertTrue(torch.allclose(out[:, 0], out[:, 8], atol=1e-5))


class Runner(unittest.TestCase):
    def test_training_is_deterministic_and_counts_guard_steps(self):
        torch.set_num_threads(1)
        m1, r1 = dr.train('perm3', 'gru', 16, seed=2, steps=4, batch=8)
        m2, r2 = dr.train('perm3', 'gru', 16, seed=2, steps=4, batch=8)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(m1.parameters(), m2.parameters())))
        self.assertEqual(r1['guard']['steps'], 4)
        self.assertEqual(r1['guard']['skipped'], 0)

    def test_curriculum_advances_only_after_the_accuracy_threshold_and_never_trains_past_the_last_length(self):
        torch.set_num_threads(1)
        _, r = dr.train('perm3', 'gru', 32, seed=1, steps=500, batch=32, curriculum=[2, 4], advance_at=0.9, min_stage_steps=50, window=20)
        self.assertEqual(r['curriculum'], [2, 4])
        self.assertGreaterEqual(len(r['advances']), 1)
        self.assertEqual(r['advances'][0]['from_length'], 2)
        self.assertEqual(r['advances'][0]['to_length'], 4)
        self.assertGreaterEqual(r['advances'][0]['step'], 50)               # never before the minimum stage length
        self.assertEqual(r['final_stage_length'], 4)
        # an impossible threshold keeps the first stage for the whole run
        _, r = dr.train('perm3', 'gru', 16, seed=1, steps=60, batch=8, curriculum=[2, 4], advance_at=1.01, min_stage_steps=5, window=5)
        self.assertEqual(r['advances'], [])
        self.assertEqual(r['final_stage_length'], 2)

    def test_evaluate_reports_accuracy_in_range_and_uses_the_requested_lengths(self):
        m = make_arm('gru', 'perm3', 16, seed=1)
        curve = dr.evaluate(m, 'perm3', [16, 32], n=10, seed=1)
        self.assertEqual(sorted(curve), [16, 32])
        for row in curve.values():
            self.assertTrue(0.0 <= row['final'] <= 1.0 and 0.0 <= row['prefix'] <= 1.0)
        self.assertIsNone(curve[16]['beyond_horizon'] if dt.TRAIN_LENGTH >= 16 else 0)   # length 16: nothing beyond the trained horizon
        self.assertIsNotNone(curve[32]['beyond_horizon'])

    def test_count_evaluation_reports_deep_and_reentry_accuracy(self):
        m = make_arm('gru', 'count', 16, seed=1)
        curve = dr.evaluate(m, 'count', [8, 16], n=10, seed=1)
        self.assertIsNone(curve[8]['deep'])                     # peak 8 = the trained range: no position is deeper than 8
        self.assertIsNone(curve[8]['reentry'])
        self.assertIsNone(curve[8]['resync'])
        self.assertIsNotNone(curve[16]['deep'])
        self.assertIsNotNone(curve[16]['reentry'])              # climbs to 16, then comes back down to <= 8: first descent
        self.assertIsNotNone(curve[16]['resync'])               # and keeps wandering after the floor reset
        self.assertGreater(curve[16]['reentry_positions'], 0)

    def test_effective_horizon_skips_points_where_a_metric_does_not_apply(self):
        curve = {8: dict(reentry=None), 16: dict(reentry=1.0), 32: dict(reentry=0.99), 64: dict(reentry=0.4)}
        self.assertEqual(dr.effective_horizon(curve, 'reentry'), 32)

    def test_curriculum_advances_are_logged_so_long_runs_can_be_followed(self):
        torch.set_num_threads(1)
        lines = []
        dr.train('perm3', 'gru', 32, seed=1, steps=500, batch=32, curriculum=[2, 4], advance_at=0.9, min_stage_steps=50, window=20, log=lines.append)
        self.assertTrue(any('curriculum advance 2 -> 4' in line for line in lines))

    def test_saved_weights_reload_to_identical_outputs_and_load_with_weights_only(self):
        torch.set_num_threads(1)
        model, trained = dr.train('perm3', 'gru', 16, seed=3, steps=3, batch=8)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'm.pt'
            dr.save_model(model, path, task='perm3', arm='gru', width=16, seed=3, trained=trained)
            raw = torch.load(path, weights_only=True)                       # plain tensors + primitives only
            self.assertEqual(raw['meta']['arm'], 'gru')
            loaded = dr.load_model(path)
        x, _ = dt.generate('perm3', 4, 12, seed=11)
        model.eval()
        loaded.eval()
        self.assertTrue(torch.equal(model(x), loaded(x)))

    def test_run_writes_the_weights_before_the_slow_evaluation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'sub' / 'r.json'
            dr.run('perm3', 'gru', 16, seed=1, steps=2, points=[16], n_eval=4, out=out, batch=8)
            self.assertTrue(out.exists())
            self.assertTrue(out.with_suffix('.pt').exists())

    def test_first_error_hazard_counts_first_errors_among_sequences_still_error_free(self):
        target = torch.zeros(4, 8, dtype=torch.long)
        pred = target.clone()
        pred[0, 2] = 1            # first error at 2 (and again later: only the first counts)
        pred[0, 6] = 1
        pred[1, 5] = 1            # first error at 5
        pred[3, 2] = 1            # first error at 2
        rows = dr.first_error_hazard(pred, target, [0, 4, 8])            # sequence 2 never errs
        self.assertEqual([(r['start'], r['stop'], r['at_risk'], r['first_errors']) for r in rows], [(0, 4, 4, 2), (4, 8, 2, 1)])
        self.assertAlmostEqual(rows[0]['rate'], 2 / (4 * 4))
        self.assertAlmostEqual(rows[1]['rate'], 1 / (2 * 4))

    def test_first_error_hazard_perfect_predictions_and_window_clipping(self):
        target = torch.zeros(3, 10, dtype=torch.long)
        rows = dr.first_error_hazard(target.clone(), target, [0, 4, 8, 16, 32])
        self.assertEqual([(r['start'], r['stop']) for r in rows], [(0, 4), (4, 8), (8, 10)])   # windows past the end are dropped, the last is clipped
        self.assertTrue(all(r['first_errors'] == 0 and r['rate'] == 0.0 and r['at_risk'] == 3 for r in rows))

    def test_first_error_hazard_rate_is_none_when_nobody_is_left(self):
        target = torch.zeros(2, 8, dtype=torch.long)
        pred = target.clone()
        pred[:, 0] = 1
        rows = dr.first_error_hazard(pred, target, [0, 4, 8])
        self.assertEqual(rows[0]['first_errors'], 2)
        self.assertEqual(rows[1]['at_risk'], 0)
        self.assertIsNone(rows[1]['rate'])

    def test_permutation_evaluation_reports_the_hazard_windows(self):
        m = make_arm('gru', 'perm3', 16, seed=1)
        curve = dr.evaluate(m, 'perm3', [16, 64], n=10, seed=1)
        self.assertEqual([(r['start'], r['stop']) for r in curve[16]['hazard']], [(0, 16)])
        self.assertEqual([(r['start'], r['stop']) for r in curve[64]['hazard']], [(0, 16), (16, 32), (32, 64)])

    def test_length_control_uses_the_training_distribution_at_long_lengths(self):
        m = make_arm('gru', 'count', 16, seed=1)
        control = dr.evaluate_length_control(m, [64, 128], n=6, seed=1)
        self.assertEqual(sorted(control), [64, 128])
        for row in control.values():
            self.assertTrue(0.0 <= row['final'] <= 1.0 and 0.0 <= row['prefix'] <= 1.0)

    def test_an_untrained_model_is_flagged_as_not_learned_and_gets_no_horizon(self):
        torch.set_num_threads(1)
        r = dr.run('count', 'gru', 16, seed=1, steps=0, points=[8, 16], n_eval=6)
        self.assertFalse(r['learned_in_range'])
        self.assertEqual(r['effective_horizon'], 0)
        self.assertEqual(sorted(r['length_control']), [64, 256, 1024])

    def test_counter_trains_on_long_sequences_so_only_depth_is_extrapolated(self):
        self.assertEqual(dt.train_length('count'), dt.COUNT_TRAIN_LENGTH)
        self.assertEqual(dt.train_length('perm5'), dt.TRAIN_LENGTH)
        b = dt.train_batch(dr.NAMESPACE, 'count', 1, 1, 8)
        self.assertEqual(b['x'].shape[1], dt.COUNT_TRAIN_LENGTH)
        self.assertLessEqual(int(b['y'].max()), dt.TRAIN_DEPTH)
        # the shallowest evaluation point (peak = the trained peak) must use a sequence length that was trained on
        m = make_arm('gru', 'count', 16, seed=1)
        curve = dr.evaluate(m, 'count', [8], n=4, seed=1)
        self.assertEqual(curve[8]['length'], dt.COUNT_TRAIN_LENGTH)

    def test_eval_split_is_separate_from_training_data(self):
        b = dt.train_batch(dr.NAMESPACE, 'perm4', 1, 1, 8)
        x, _, _ = dt.heldout(dr.NAMESPACE, 'perm4', 1, 'dev', 8, dt.TRAIN_LENGTH)
        self.assertFalse(torch.equal(b['x'], x))

    def test_effective_horizon(self):
        curve = {16: dict(final=1.0), 32: dict(final=0.97), 64: dict(final=0.90), 128: dict(final=1.0)}
        self.assertEqual(dr.effective_horizon(curve, 'final'), 32)       # 64 breaks the run; the 128 recovery does not count
        self.assertEqual(dr.effective_horizon({16: dict(final=0.5)}, 'final'), 0)
        self.assertEqual(dr.effective_horizon({8: dict(unseen_depth=None)}, 'unseen_depth'), 0)

    def test_exactly_the_threshold_meets_it_although_float32_stores_it_just_below(self):
        # 95 of 100 sequences right: the float32 mean is 0.949999988..., which a plain `>= 0.95` rejects (the S4 LSTM w112 pilot: its horizon
        # stopped at 64 although 128 and 256 both had exactly 95/100)
        value = float(torch.tensor([1.0] * 95 + [0.0] * 5).mean())
        self.assertLess(value, dr.THRESHOLD)
        self.assertTrue(dr.meets(value))
        self.assertEqual(dr.effective_horizon({16: dict(final=1.0), 32: dict(final=value), 64: dict(final=value), 128: dict(final=0.94)}, 'final'), 64)
        self.assertFalse(dr.meets(0.9499))                               # the tolerance covers float32 rounding only


class LearningRateDecay(unittest.TestCase):
    def test_zero_is_the_previous_constant_rate_bit_for_bit_and_a_fraction_decays_the_last_updates(self):
        constant, a = dr.train('perm3', 'gru', 16, seed=1, steps=10, batch=8, curriculum=[4], lr=0.01)
        also, b = dr.train('perm3', 'gru', 16, seed=1, steps=10, batch=8, curriculum=[4], lr=0.01, lr_decay_fraction=0.0)
        decayed, c = dr.train('perm3', 'gru', 16, seed=1, steps=10, batch=8, curriculum=[4], lr=0.01, lr_decay_fraction=0.5)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(constant.parameters(), also.parameters())))
        self.assertEqual((a['lr_last'], b['lr_last']), (0.01, 0.01))
        self.assertAlmostEqual(c['lr_last'], 0.01 * 1 / 5)                   # step 10 of 10, decay over the last 5: (10 - 10 + 1) / 5
        self.assertEqual(c['lr_decay_fraction'], 0.5)
        self.assertFalse(all(torch.equal(p, q) for p, q in zip(constant.parameters(), decayed.parameters())))

    def test_the_last_update_of_a_30_percent_decay_runs_at_one_thirtieth_of_the_rate(self):
        _, info = dr.train('perm3', 'gru', 8, seed=1, steps=100, batch=4, curriculum=[3], lr=0.03, lr_decay_fraction=0.3)
        self.assertAlmostEqual(info['lr_last'], 0.03 / 30)                  # the window is the last 30 updates (71 .. 100): the last one has (100 - 100 + 1) / 30 of the rate

    def test_the_floor_lifts_the_end_of_the_decay_and_is_ignored_without_a_decay(self):
        base = dict(seed=1, steps=10, batch=8, curriculum=[4], lr=0.01)
        model0, a = dr.train('perm3', 'gru', 16, lr_decay_fraction=0.5, **base)
        model_explicit, b = dr.train('perm3', 'gru', 16, lr_decay_fraction=0.5, lr_floor=0.0, **base)
        _, c = dr.train('perm3', 'gru', 16, lr_decay_fraction=0.5, lr_floor=0.1, **base)
        _, d = dr.train('perm3', 'gru', 16, lr_floor=0.1, **base)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(model0.parameters(), model_explicit.parameters())))     # floor 0 is the E-P behaviour, bit for bit
        self.assertAlmostEqual(a['lr_last'], 0.01 * 0.2)
        self.assertAlmostEqual(c['lr_last'], 0.01 * (0.1 + 0.9 * 0.2))
        self.assertEqual((d['lr_last'], d['lr_floor']), (0.01, 0.1))         # without a decay the floor changes nothing


if __name__ == '__main__':
    unittest.main()
