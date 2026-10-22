import unittest

import torch

from horizon import erm_models as em
from horizon import erm_r as er
from horizon import erm_tasks as et


class TaskTests(unittest.TestCase):
    def test_distinct_keys_give_a_perfect_memory_the_whole_score(self):
        x, apos, tgt, dist = er.generate(300, seed=5)
        self.assertEqual(er.ceiling(x, apos, tgt), 1.0)

    def test_the_sealed_generator_is_untouched_and_its_ceiling_is_about_0_70(self):
        x, apos, tgt, _ = et.generate(400, seed=777)
        c = er.ceiling(x, apos[:, None], tgt[:, None])
        self.assertGreater(c, 0.62)
        self.assertLess(c, 0.78)

    def test_layout(self):
        x, apos, tgt, dist = er.generate(50, seed=1)
        self.assertEqual(tuple(apos.shape), (50, er.QUERIES))
        fact_part = x[:, :3 * et.WINDOW]
        for s in range(50):
            keys = [int(t) for t in fact_part[s] if int(t) in et.KEYS]
            self.assertEqual(len(keys), et.N_FACTS)
            self.assertEqual(len(set(keys)), et.N_FACTS)                      # distinct keys
            self.assertTrue(bool((apos[s] >= 3 * et.WINDOW).all()))          # every query in the last window
            self.assertEqual(len(set(x[s, apos[s]].tolist())), er.QUERIES)  # about distinct facts
            for j in range(er.QUERIES):
                self.assertEqual(int(x[s, apos[s, j] - 1]), et.QMARK)
                self.assertNotIn(int(tgt[s, j]), x[s, 3 * et.WINDOW:].tolist())   # the value is not in the query window
        self.assertTrue(set(dist.unique().tolist()) <= {1, 2, 3})

    def test_answers_pick_the_answer_positions_for_every_arm(self):
        x, apos, tgt, _ = er.generate(4, seed=2)
        for arm in ('A', 'Apr', 'B'):
            model = em.make_arm(arm, seed=1)
            sel = er.answers(model(x.reshape(4, et.N_WINDOWS, et.WINDOW)), apos)
            self.assertEqual(tuple(sel.shape), (4, er.QUERIES, et.VOCAB), arm)

    def test_a_short_curriculum_run_and_the_reading(self):
        model, info = er.train('Apr', 1, 6, batch=4, curriculum=[4, 8, 16], log=lambda *_: None)
        self.assertEqual(info['curriculum'], [4, 8, 16])
        ev = er.evaluate(model, n=20)
        self.assertEqual(set(ev), {'all', '1', '2', '3'})
        r = er.reading({'Apr': dict(eval={'3': dict(acc=0.95)}), 'A': dict(eval={'3': dict(acc=0.12)}),
                        'B': dict(eval={'3': dict(acc=0.5)})})
        self.assertEqual((r['positive_control'], r['control'], r['B']), ('LEARNABLE', 'CONTROL OK', 'PARTIAL'))
        r = er.reading({'Apr': dict(eval={'3': dict(acc=0.4)}), 'B': dict(eval={'3': dict(acc=0.95)})})
        self.assertTrue(r['B'].startswith('VACUOUS'))


if __name__ == '__main__':
    unittest.main()
