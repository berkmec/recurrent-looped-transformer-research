"""E-S state-keyed retrieval: the sequences follow the layout, the value labels are the fact keyed by the CURRENT state and the state labels the running product (both
checked against an independent replay), the runner trains and evaluates with the two heads, the summary reads the right buckets."""
import tempfile
import unittest
from pathlib import Path

import torch

from horizon import depth_tasks as dt
from horizon import sk_run as run
from horizon import sk_task as sk


def replay(n, row):
    """Independent re-implementation: parse facts, walk the group, return (facts, [(query position, expected value, walk tokens before)], {position: state})."""
    g = sk.size(n)
    table = dt.group(n)[2]
    facts, i = {}, 0
    while i < len(row) and g <= row[i] < 2 * g:                              # fact keys come first, each followed by its value
        facts[row[i] - g] = row[i + 1] - 2 * g
        i += 2
    state, walked, queries, states = 0, 0, [], {}
    for t in range(i, len(row)):
        if row[t] == sk.query_token(n):
            queries.append((t, facts[state], walked))
        else:
            assert 0 <= row[t] < g
            state = int(table[state, row[t]])
            walked += 1
        states[t] = state
    return facts, queries, states


class Layout(unittest.TestCase):
    def test_vocabulary_and_classes(self):
        self.assertEqual((dt.vocab('sk3'), dt.classes('sk3')), (2 * 6 + 8 + 1, 6 + 8))
        self.assertEqual((dt.vocab('sk4'), dt.classes('sk4')), (2 * 24 + 8 + 1, 24 + 8))
        with self.assertRaises(ValueError):
            dt.generate('sk3', 2, 4, 0)
        with self.assertRaises(ValueError):
            dt.oracle('sk3', torch.zeros(1, 4, dtype=torch.long))

    def test_value_and_state_labels_follow_an_independent_replay(self):
        for n in (3, 4):
            b = sk.generate(n, 20, 9, seed=5, queries=3)
            for i in range(b['x'].shape[0]):
                facts, queries, states = replay(n, b['x'][i].tolist())
                self.assertEqual(sorted(facts), list(range(sk.size(n))))                       # every key exactly once
                self.assertEqual(len(queries), 3)
                for pos, value, walked in queries:
                    self.assertEqual(int(b['value'][i, pos]), value)
                    self.assertEqual(int(b['walk'][i, pos]), walked)
                self.assertEqual(int((b['value'][i] != sk.IGNORE).sum()), 3)                   # the value head is supervised at the queries only
                self.assertEqual(queries[-1][2], 9)                                            # the last query is always after the whole walk
                for t, state in states.items():                                                 # the state head at every walk and query position
                    self.assertEqual(int(b['state'][i, t]), state)
                self.assertTrue(bool((b['state'][i, :2 * sk.size(n)] == sk.IGNORE).all()))     # nothing inside the facts
                self.assertTrue(bool((b['walk'][i, :2 * sk.size(n)] == -1).all()))

    def test_walk_zero_asks_for_the_identity_key_and_values_differ_between_sequences(self):
        b = sk.generate(3, 30, 0, seed=1)
        x = b['x']
        self.assertEqual(x.shape[1], 2 * 6 + 1)
        for i in range(30):
            facts, queries, _ = replay(3, x[i].tolist())
            self.assertEqual(queries, [(12, facts[0], 0)])
            self.assertEqual(int(b['value'][i, 12]), facts[0])
            self.assertEqual(int(b['state'][i, 12]), 0)
        tables = {tuple(replay(3, x[i].tolist())[0].items()) for i in range(30)}
        self.assertGreater(len(tables), 20)                                                    # facts are redrawn for every sequence

    def test_the_evaluation_layout_asks_after_every_k_steps_and_after_the_last_one(self):
        b = sk.generate(3, 10, 10, seed=2, every=4)
        for i in range(10):
            self.assertEqual([w for _, _, w in replay(3, b['x'][i].tolist())[1]], [4, 8, 10])
        b = sk.generate(3, 5, 8, seed=2, every=4)
        for i in range(5):
            self.assertEqual([w for _, _, w in replay(3, b['x'][i].tolist())[1]], [4, 8])

    def test_determinism_and_batch_shape(self):
        a = sk.generate(4, 6, 12, seed=7, queries=4)
        b = sk.generate(4, 6, 12, seed=7, queries=4)
        c = sk.generate(4, 6, 12, seed=8, queries=4)
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        self.assertFalse(torch.equal(a['x'], c['x']))
        self.assertEqual(a['x'].shape, (6, 2 * 24 + 12 + 4))
        with self.assertRaises(ValueError):
            sk.generate(3, 0, 4, 0)

    def test_buckets_and_accuracies(self):
        self.assertEqual([sk.bucket(v) for v in (0, 3, 4, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 700)],
                         ['0-3', '0-3', '4-15', '4-15', '16-31', '16-31', '32-63', '32-63', '64-127', '64-127', '128-255', '128-255', '256-511', '256-511', '512+', '512+'])
        value = torch.tensor([[sk.IGNORE, 1, sk.IGNORE, 2], [sk.IGNORE, 3, sk.IGNORE, 4]])
        pred = torch.tensor([[0, 1, 0, 9], [0, 3, 0, 4]])
        walk = torch.tensor([[-1, 4, -1, 40], [-1, 4, -1, 40]])
        self.assertEqual(sk.value_accuracy(pred, value, walk), {'all': [3, 4], '4-15': [2, 2], '32-63': [1, 2]})
        state = torch.tensor([[sk.IGNORE, 5, 6, 7]])
        self.assertEqual(sk.state_accuracy(torch.tensor([[0, 5, 0, 7]]), state, torch.tensor([[-1, 1, 2, 70]])), {'all': [2, 3], '0-3': [1, 2], '64-127': [1, 1]})


class Runner(unittest.TestCase):
    def test_train_and_evaluate_smoke_for_a_recurrent_arm_and_the_core(self):
        for arm, width in (('gru', 16), ('core', 32)):
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / 'm.json'
                r = run.run('sk3', arm, width, 1, steps=3, walks=(4, 8), n_eval=4, out=out, batch=4, curriculum=(0, 1), log_every=0)
                self.assertTrue(out.exists() and out.with_suffix('.pt').exists())
                self.assertEqual(sorted(r['curve']), ['4', '8'])
                self.assertEqual(r['curve']['8']['queries'], 4 * 2)                            # queries after walk tokens 4 and 8
                self.assertIn('in_range', r)
                self.assertIn('state_extrapolation', r)
                self.assertEqual(r['train']['curriculum'], [0, 1])

    def test_the_loss_has_two_heads_and_ignores_unsupervised_positions(self):
        g = sk.size(3)
        b = sk.generate(3, 4, 3, seed=1, queries=2)
        logits = torch.zeros(4, b['x'].shape[1], dt.classes('sk3'))
        loss, acc = run.loss_and_accuracy(logits, b, g)
        self.assertAlmostEqual(float(loss), float(torch.log(torch.tensor(6.0)) + torch.log(torch.tensor(8.0))), places=5)   # uniform logits: ln G + ln V
        perfect = torch.zeros_like(logits)
        for i in range(4):
            for t in range(b['x'].shape[1]):
                if b['state'][i, t] != sk.IGNORE:
                    perfect[i, t, int(b['state'][i, t])] = 30.0
                if b['value'][i, t] != sk.IGNORE:
                    perfect[i, t, g + int(b['value'][i, t])] = 30.0
        loss, acc = run.loss_and_accuracy(perfect, b, g)
        self.assertLess(float(loss), 1e-6)
        self.assertEqual(acc, 1.0)

    def test_learning_rate_is_annealed_for_every_arm_by_default(self):
        _, info = run.train('sk3', 'gru', 16, 1, steps=20, batch=4, curriculum=(0, 1))
        self.assertAlmostEqual(info['lr_last'], 3e-3 / (0.3 * 20))                         # the last update: (20 - 20 + 1) / (0.3 x 20) of the base rate

    def test_the_summary_reads_the_walk_16_accuracy_and_the_two_extrapolation_buckets_of_the_longest_walk(self):
        curve = {16: dict(accuracy=0.9, state_accuracy=0.99, buckets={'all': [9, 10]}),
                 512: dict(accuracy=0.4, state_accuracy=0.8, buckets={'all': [40, 100], '32-63': [8, 10], '64-127': [6, 30], '128-255': [26, 60]},
                           state_buckets={'all': [800, 1000], '32-63': [90, 100], '64-127': [180, 200], '128-255': [530, 700]})}
        s = run.summary(curve)
        self.assertEqual((s['in_range'], s['state_in_range']), (0.9, 0.99))
        self.assertAlmostEqual(s['extrapolation'], 14 / 40)
        self.assertAlmostEqual(s['state_extrapolation'], 270 / 300)
        self.assertEqual(s['extrapolation_queries'], 40)

    def result(self, arm, seed, in_range, ext, state_ext=0.99):
        return dict(arm=arm, seed=seed, in_range=in_range, extrapolation=ext, state_extrapolation=state_ext)

    def read(self, spec, **kw):
        """spec: {arm: [(in_range, extrapolation), ...]}"""
        return run.reading([self.result(arm, i + 1, *v) for arm, vs in spec.items() for i, v in enumerate(vs)], **kw)

    def test_reading_advantage_no_advantage_not_learned_and_mixed(self):
        both = {'core': [(0.99, 0.80), (0.98, 0.76)], 'gru': [(0.97, 0.30), (0.96, 0.34)], 'tfm': [(0.60, 0.20), (0.62, 0.22)]}
        r = self.read(both)
        self.assertEqual(r['verdict'], 'ADVANTAGE')                                      # 0.78 >= 0.6 and >= 0.32 + 0.15; the unlearned tfm does not count as a rival
        self.assertAlmostEqual(r['best_other_extrapolation'], 0.32)
        self.assertEqual(self.read({**both, 'lstm': [(0.99, 0.75), (0.99, 0.77)]})['verdict'], 'NO ADVANTAGE')          # a learned rival within 0.05 of the Core
        self.assertEqual(self.read({'core': [(0.80, 0.9), (0.85, 0.9)], 'gru': [(0.99, 0.2), (0.99, 0.2)]})['verdict'], 'CORE NOT LEARNED')
        self.assertEqual(self.read({'core': [(0.99, 0.7), (0.99, 0.7)], 'gru': [(0.99, 0.5), (0.99, 0.5)]})['verdict'], 'ADVANTAGE')   # ahead by 0.2 >= 0.15
        self.assertEqual(self.read({'core': [(0.99, 0.60), (0.99, 0.60)], 'gru': [(0.99, 0.50), (0.99, 0.50)]})['verdict'], 'MIXED')    # ahead by 0.10: neither
        self.assertEqual(self.read({'core': [(0.99, 0.8), (0.99, 0.8)], 'gru': [(0.7, 0.9), (0.7, 0.9)], 'tfm': [(0.5, 0.1), (0.5, 0.1)]})['verdict'],
                         'ADVANTAGE (LEARNABILITY)')                                      # the rivals fail inside the trained range
        self.assertEqual(self.read({'gru': [(0.99, 0.5), (0.99, 0.5)]})['verdict'], 'NO CORE RUN')

    def test_a_verdict_is_provisional_with_fewer_than_two_seeds(self):
        r = self.read({'core': [(0.99, 0.8)], 'gru': [(0.99, 0.3), (0.99, 0.3)]})
        self.assertIn('provisional', r['verdict'])
        self.assertTrue(r['verdict'].startswith('ADVANTAGE'))

    def test_every_arm_accepts_the_task(self):
        from horizon.depth_models import ARMS, make_arm
        b = sk.generate(3, 3, 5, seed=1, queries=2)
        for arm in ARMS:
            m = make_arm(arm, 'sk3', 32, 1)
            self.assertEqual(tuple(m(b['x']).shape), (3, b['x'].shape[1], dt.classes('sk3')), arm)


if __name__ == '__main__':
    unittest.main()
