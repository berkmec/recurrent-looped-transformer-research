"""E-A9b: the k-means finds well-separated clusters, the purity is what it says, the registered reading is mechanical, a tiny end-to-end run has every part."""
import tempfile
import unittest
from pathlib import Path

import torch

from horizon import depth_run as dr
from horizon import snap_labelfree as sf


class Clusters(unittest.TestCase):
    def test_kmeans_recovers_well_separated_clusters(self):
        g = torch.Generator().manual_seed(0)
        centers = 10.0 * torch.randn(6, 8, generator=g)
        labels = torch.arange(6).repeat_interleave(100)
        points = centers[labels] + 0.3 * torch.randn(600, 8, generator=g)
        centroids, assignment = sf.kmeans(points, 6)
        self.assertEqual(sf.purity(assignment, labels, 6), 1.0)
        self.assertEqual(len(torch.unique(assignment)), 6)
        # every true centre has a centroid within the noise radius
        self.assertLess(float(torch.cdist(centers, centroids).min(1).values.max()), 0.5)

    def test_purity_counts_the_majority_label_of_each_cluster(self):
        assignment = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1])
        self.assertAlmostEqual(sf.purity(assignment, labels, 2), (3 + 3) / 8)


def fake(task='perm5', width=128, base=(100, 10000), primary=None, accuracy=1.0, stored=True):
    primary = primary if primary is not None else base
    conditions = {'base': dict(events=base[0], person_steps=base[1]), sf.PRIMARY: dict(events=primary[0], person_steps=primary[1]), 'kmeans-1': dict(events=base[0], person_steps=base[1])}
    return dict(weights=f'x/{task}-w{width}.pt', task=task, width=width, seed=1, n=200, length=1024, parameters=1, clusters=1, purity=1.0, conditions=conditions,
                in_range={sf.PRIMARY: accuracy, 'kmeans-1': 1.0}, stored_control=stored)


class Reading(unittest.TestCase):
    def s5(self, **kw):
        return [fake(width=128 + i, **kw) for i in range(3)]

    def test_verdicts_follow_the_registered_rule(self):
        s4 = [fake('perm4', 32, base=(0, 10000), primary=(0, 10000))]
        self.assertEqual(sf.reading(self.s5(primary=(10, 10000)) + s4)['verdict'], 'LABEL-FREE SNAP WORKS')
        self.assertEqual(sf.reading(self.s5() + s4)['verdict'], 'LABEL-FREE SNAP DOES NOT WORK')
        self.assertEqual(sf.reading([fake(width=128, primary=(10, 10000)), fake(width=129), fake(width=130)] + s4)['verdict'], 'MIXED')
        self.assertTrue(sf.reading(self.s5(accuracy=0.9) + s4)['verdict'].startswith('INCONCLUSIVE'))
        self.assertTrue(sf.reading(self.s5() + [fake('perm4', 32, base=(0, 10000), primary=(30, 10000))])['verdict'].startswith('INCONCLUSIVE'))
        self.assertTrue(sf.reading(self.s5(stored=False) + s4)['verdict'].startswith('CONTROL FAILED'))
        self.assertEqual(sf.reading(s4)['verdict'], 'NO S5 MODEL')


class EndToEnd(unittest.TestCase):
    def test_a_tiny_run_has_every_part(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'm.json'
            dr.run('perm3', 'core', 32, 1, steps=2, points=[16], n_eval=4, out=out, batch=4, curriculum=[4])
            r = sf.measure(out.with_suffix('.pt'), n=6, length=60, batch=3, log=lambda m: None)
        self.assertEqual((r['task'], r['clusters']), ('perm3', 6))
        self.assertEqual(set(r['conditions']), {'base', 'kmeans-16', 'kmeans-1'})
        self.assertEqual(set(r['in_range']), {'kmeans-16', 'kmeans-1'})
        self.assertGreater(r['purity'], 0.0)
        self.assertIsNone(r['stored_control'])
        self.assertIn('kmeans-16', sf.describe(r))
        self.assertEqual(set(sf.labels_of(r)), {'kmeans-16', 'kmeans-1'})


if __name__ == '__main__':
    unittest.main()
