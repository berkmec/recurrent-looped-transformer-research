"""E-A10p: the retrofit sets the prototypes and the gate as documented, a closed module is the untouched model, the registered reading is mechanical, a tiny end-to-end run has every part."""
import math
import tempfile
import unittest
from pathlib import Path

import torch

from horizon import cleanup as cl
from horizon import cleanup_probe as cp
from horizon import depth_models as dm
from horizon import depth_run as dr


def tiny(seed=3):
    return dm.make_arm('core', 'perm3', 32, seed).eval()


def tokens(batch=3, length=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 6, (batch, length), generator=g)


class Retrofit(unittest.TestCase):
    def test_the_prototypes_the_temperature_and_the_gate_are_set_and_the_original_is_untouched(self):
        model = tiny()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        r, labels = cp.feedback_vectors(model, 'perm3', 1, n=300)
        protos = cp.labelled_prototypes(r, labels, 6)
        out = cp.retrofit(model, protos)
        self.assertIsInstance(out.pn, cl.CleanupNorm)
        self.assertAlmostEqual(cl.gate_value(out), 0.999, places=4)
        self.assertAlmostEqual(float(out.pn.log_beta.detach().exp()), cp.BETA, places=3)
        self.assertTrue(torch.equal(out.pn.keys[:6], protos))
        self.assertTrue(torch.equal(out.pn.keys[6:12], protos))                                  # the other slots repeat the prototypes
        self.assertTrue(torch.equal(out.pn.keys, out.pn.values))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))   # deep copy: the model that was passed in is unchanged
        self.assertFalse(isinstance(model.pn, cl.CleanupNorm))

    def test_a_closed_module_is_the_untouched_model_and_an_open_one_is_not(self):
        model, x = tiny(), tokens()
        r, labels = cp.feedback_vectors(model, 'perm3', 1, n=300)
        protos = cp.labelled_prototypes(r, labels, 6)
        with torch.no_grad():
            base = model(x)
            self.assertTrue(torch.equal(cp.retrofit(model, protos, gate=0.0)(x), base))
            self.assertFalse(torch.allclose(cp.retrofit(model, protos)(x)[:, 4:], base[:, 4:], atol=1e-4))

    def test_labelled_prototypes_need_enough_states_per_class(self):
        with self.assertRaises(ValueError):
            cp.labelled_prototypes(torch.zeros(10, 4), torch.zeros(10, dtype=torch.long), 3)


def fake(task='perm5', width=128, base=(100, 10000), labelled=None, in_range=1.0, identity=True, stored=True):
    labelled = labelled if labelled is not None else base
    conditions = {'base': dict(events=base[0], person_steps=base[1], in_range=1.0), 'labelled': dict(events=labelled[0], person_steps=labelled[1], in_range=in_range),
                  'kmeans': dict(events=base[0], person_steps=base[1], in_range=1.0)}
    return dict(weights=f'x/{task}-w{width}.pt', task=task, width=width, seed=1, n=200, length=1024, conditions=conditions, gate0_identity=identity, kmeans_clusters=120, stored_control=stored)


class Reading(unittest.TestCase):
    def test_verdicts_follow_the_registered_rule(self):
        self.assertEqual(cp.reading([fake(width=128, labelled=(10, 10000)), fake(width=129, labelled=(10, 10000))])['verdict'], 'THE MODULE CAN CLEAN')
        self.assertEqual(cp.reading([fake(width=128), fake(width=129)])['verdict'], 'THE MODULE CANNOT CLEAN')
        self.assertEqual(cp.reading([fake(width=128, labelled=(10, 10000)), fake(width=129)])['verdict'], 'MIXED')
        self.assertEqual(cp.reading([fake(width=128, labelled=(40, 10000)), fake(width=129, labelled=(40, 10000))])['verdict'], 'MIXED')               # PARTIAL is neither
        self.assertEqual(cp.reading([fake(width=128, labelled=(10, 10000), in_range=0.9), fake(width=129, labelled=(10, 10000))])['verdict'], 'MIXED')    # the module lowers the in-range accuracy
        self.assertTrue(cp.reading([fake(identity=False), fake(width=129)])['verdict'].startswith('CONTROL FAILED'))
        self.assertTrue(cp.reading([fake(stored=False), fake(width=129)])['verdict'].startswith('CONTROL FAILED'))
        self.assertEqual(cp.reading([fake('perm4')])['verdict'], 'NO S5 MODEL')


class EndToEnd(unittest.TestCase):
    def test_a_tiny_run_has_every_part(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'core', 32, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            r = cp.measure(out.with_suffix('.pt'), n=6, length=60, batch=3, log=lambda m: None)
        self.assertEqual(set(r['conditions']), {'base', 'labelled', 'kmeans'})
        self.assertTrue(r['gate0_identity'])
        self.assertIsNone(r['stored_control'])
        self.assertGreater(r['kmeans_clusters'], 0)
        self.assertIn('labelled', cp.describe(r))
        self.assertEqual(set(cp.labels_of(r)), {'labelled', 'kmeans'})


if __name__ == '__main__':
    unittest.main()
