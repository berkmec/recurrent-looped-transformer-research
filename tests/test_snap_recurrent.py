"""E-A9c: the unrolled recurrence equals the fused one, the snap equals an independent cell-by-cell implementation, the registered reading is mechanical, a tiny end-to-end run has every part."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from horizon import depth_models as dm
from horizon import depth_run as dr
from horizon import depth_tasks as dt
from horizon import snap_recurrent as sr
from horizon.state_growth import recurrent_trajectory


def arm(kind, seed=3):
    return dm.make_arm(kind, 'perm3', 16, seed).eval()


def tokens(batch=3, length=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 6, (batch, length), generator=g)


def cell_snap(a, x, centroids, selector, every):
    """An independent implementation with torch's own cells."""
    lstm = isinstance(a.rnn, nn.LSTM)
    width = a.rnn.hidden_size
    cell = nn.LSTMCell(width, width) if lstm else nn.GRUCell(width, width)
    cell.weight_ih.data.copy_(a.rnn.weight_ih_l0.data)
    cell.weight_hh.data.copy_(a.rnn.weight_hh_l0.data)
    cell.bias_ih.data.copy_(a.rnn.bias_ih_l0.data)
    cell.bias_hh.data.copy_(a.rnn.bias_hh_l0.data)
    emb = a.embedding(x)
    h, c, out = emb.new_zeros(len(x), width), emb.new_zeros(len(x), width), []
    for t in range(x.shape[1]):
        if t >= 1 and t % every == 0:
            state = torch.cat([h, c], -1) if lstm else h
            chosen = a.read(h).argmax(-1) if selector == 'class' else torch.cdist(state, centroids).argmin(-1)
            h = centroids[chosen][:, :width]
            if lstm:
                c = centroids[chosen][:, width:]
        if lstm:
            h, c = cell(emb[:, t], (h, c))
        else:
            h = cell(emb[:, t], h)
        out.append(h)
    return a.read(torch.stack(out, 1))


class Unrolled(unittest.TestCase):
    def test_the_unrolled_recurrence_equals_the_fused_forward_for_both_arms(self):
        x = tokens()
        for kind in ('gru', 'lstm'):
            a = arm(kind)
            with torch.no_grad():
                self.assertLess(float((sr.unrolled_logits(a, x) - a(x)).abs().max()), 1e-5, kind)

    def test_the_snap_equals_an_independent_cell_implementation(self):
        x = tokens(3, 26)
        for kind, width in (('gru', 16), ('lstm', 32)):
            a = arm(kind)
            g = torch.Generator().manual_seed(1)
            centroids = torch.randn(6, width, generator=g)
            for selector, every in (('class', 1), ('class', 4), ('nearest', 3)):
                with torch.no_grad():
                    mine = sr.unrolled_logits(a, x, centroids, selector, every)
                    theirs = cell_snap(a, x, centroids, selector, every)
                self.assertLess(float((mine - theirs).abs().max()), 1e-5, (kind, selector, every))

    def test_class_centroids_are_the_class_means_of_the_state(self):
        a = arm('lstm')
        c = sr.class_centroids(a, 'perm3', 1, n=400, length=16, positions=(8, 16))
        self.assertEqual(tuple(c.shape), (6, 32))                                         # h and c concatenated
        x, y, _ = dt.heldout(dr.NAMESPACE, 'perm3', 1, 'dev', 400, 16)
        with torch.no_grad():
            states, _ = recurrent_trajectory(a, x)
        members = states[:, 8:16].reshape(-1, 32)[y[:, 8:16].reshape(-1) == 4]
        self.assertTrue(torch.allclose(c[4], members.mean(0), atol=1e-5))
        with self.assertRaises(ValueError):
            sr.class_centroids(a, 'perm3', 1, n=2, length=16)


def fake(arm_name='gru', width=256, base=(100, 10000), primary=None, accuracy=1.0, identity=0.0, stored=True):
    primary = primary if primary is not None else base
    conditions = {'base': dict(events=base[0], person_steps=base[1]), sr.PRIMARY: dict(events=primary[0], person_steps=primary[1])}
    return dict(weights=f'x/{arm_name}-{width}.pt', arm=arm_name, task='perm5', width=width, seed=1, n=200, length=1024, parameters=1, identity=identity, conditions=conditions,
                in_range={sr.PRIMARY: accuracy}, stored_control=stored)


class Reading(unittest.TestCase):
    def four(self, **kw):
        return [fake('gru', 256, **kw), fake('gru', 257, **kw), fake('lstm', 208, **kw), fake('lstm', 209, **kw)]

    def test_verdicts_follow_the_registered_rule(self):
        self.assertEqual(sr.reading(self.four(primary=(10, 10000)))['verdict'], 'THE SNAP HELPS THE RECURRENT BASELINES TOO')
        self.assertEqual(sr.reading(self.four())['verdict'], 'THE SNAP DOES NOT HELP THE RECURRENT BASELINES')
        self.assertEqual(sr.reading([fake(primary=(10, 10000))] + self.four()[1:])['verdict'], 'MIXED')
        self.assertTrue(sr.reading(self.four(accuracy=0.9))['verdict'].startswith('INCONCLUSIVE'))
        self.assertTrue(sr.reading(self.four(identity=1e-2))['verdict'].startswith('CONTROL FAILED'))
        self.assertTrue(sr.reading(self.four(stored=False))['verdict'].startswith('CONTROL FAILED'))
        self.assertEqual(sr.reading([])['verdict'], 'NO MODELS')


class EndToEnd(unittest.TestCase):
    def test_a_tiny_run_has_every_part(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'gru', 16, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            r = sr.measure(out.with_suffix('.pt'), n=6, length=60, batch=3, log=lambda m: None)
        self.assertEqual((r['arm'], r['task'], r['width']), ('gru', 'perm3', 16))
        self.assertLess(r['identity'], 1e-4)
        self.assertEqual(set(r['conditions']), {'base', 'class-16', 'class-1', 'nearest-16'})
        self.assertEqual(set(r['in_range']), {'class-16', 'class-1', 'nearest-16'})
        self.assertIsNone(r['stored_control'])
        self.assertIn('class-16', sr.describe(r))

    def test_a_core_model_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'c.json'
            dr.run('perm3', 'core', 32, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            with self.assertRaises(ValueError):
                sr.measure(out.with_suffix('.pt'), n=4, length=40, log=lambda m: None)


if __name__ == '__main__':
    unittest.main()
