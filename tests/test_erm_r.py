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

    def test_positions_add_no_parameters_and_tell_positions_apart(self):
        for arm in ('A', 'Apr', 'B', 'B_comp'):
            plain, posed = er.make(arm, seed=1), er.make(arm, seed=1, positions=True)
            self.assertEqual(sum(p.numel() for p in plain.parameters()), sum(p.numel() for p in posed.parameters()), arm)
        emb = er.make('Apr', seed=1, positions=True).enc.embedding
        x = torch.full((1, 6), 30)                                   # the same filler token six times
        e = emb(x)[0]
        self.assertGreater(float((e[1] - e[0]).abs().max()), 0.1)    # the same token at two positions now differs
        self.assertTrue(torch.equal(er.make('Apr', seed=1).enc.embedding(x)[0][0], er.make('Apr', seed=1).enc.embedding(x)[0][5]))

    def test_fact_cells_point_at_the_values_and_the_old_outputs_are_unchanged(self):
        a = er.generate(20, seed=4)
        x, apos, tgt, dist, cells = er.generate(20, seed=4, with_facts=True)
        for u, v in zip(a, (x, apos, tgt, dist)):
            self.assertTrue(torch.equal(u, v))                                  # with_facts adds, changes nothing
        for s in range(20):
            for w, p in cells[s].tolist():
                self.assertIn(int(x[s, w * et.WINDOW + p]), et.VALS)
                self.assertIn(int(x[s, w * et.WINDOW + p - 1]), et.KEYS)

    def test_the_oracle_and_rmt_arms_run_with_positions(self):
        x, apos, tgt, dist, cells = er.generate(3, seed=2, with_facts=True)
        xw = x.reshape(3, et.N_WINDOWS, et.WINDOW)
        orc = er.make('B_orc', seed=1, positions=True)
        self.assertEqual(tuple(er.answers(orc(xw, cells), apos).shape), (3, er.QUERIES, et.VOCAB))
        rmt = er.make('B_rmt', seed=1, positions=True)
        self.assertEqual(tuple(er.answers(rmt(xw), apos).shape), (3, er.QUERIES, et.VOCAB))
        emb = rmt.enc.embedding(xw)                                             # (b, windows, w, d): positions inside a window
        self.assertTrue(torch.allclose(emb[0, 0, 5] - rmt.enc.embedding.tok(xw)[0, 0, 5], rmt.enc.embedding.pe[5], atol=1e-6))
        model, info = er.train('B_orc', 1, 3, batch=4, positions=True, log=lambda *_: None)
        self.assertEqual(set(er.evaluate(model, n=10, arm='B_orc')), {'all', '1', '2', '3'})

    def test_gamma_init_opens_the_read_and_zero_still_closes_it(self):
        x, apos, tgt, dist, cells = er.generate(3, seed=2, with_facts=True)
        xw = x.reshape(3, et.N_WINDOWS, et.WINDOW)
        b0 = er.make('B', seed=1, positions=True)                       # gamma = 0: the read path is closed (K3)
        b1 = er.make('B', seed=1, positions=True, gamma_init=1.0)       # same weights, gate open
        self.assertTrue(bool((b1.read.gamma == 1.0).all()))
        with torch.no_grad():
            no_memory = b0.head(b0.enc.forward_windows(xw))             # B's own forward without the read
            self.assertLess(float((b0(xw) - no_memory).abs().max()), 1e-6)
            self.assertGreater(float((b1(xw) - no_memory).abs().max()), 1e-3)
        with self.assertRaises(ValueError):
            er.make('A', seed=1, gamma_init=1.0)

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
