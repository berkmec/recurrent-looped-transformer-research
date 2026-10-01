import unittest
import numpy as np
from review_cycle.training_analysis import paired_accuracy, seed_summary


class AnalysisContracts(unittest.TestCase):
    def test_pairing_and_net_gain(self):
        common = dict(inputs=np.array([[0], [1], [2], [1]]), targets=np.array([[0], [1], [0], [1]]), input_ids=np.arange(4))
        a = dict(**common, predictions=np.array([[0], [1], [1], [1]]))
        b = dict(**common, predictions=np.array([[0], [0], [0], [0]]))
        result = paired_accuracy(a, b)
        self.assertEqual((result['gained'], result['lost'], result['difference']), (2, 1, .25))
        for key in ('inputs', 'targets', 'input_ids'):
            broken = {**b, key: b[key] + 1}
            with self.assertRaisesRegex(ValueError, 'UNPAIRED'):
                paired_accuracy(a, broken)

    def test_zero_effect_and_seed_uncertainty(self):
        self.assertEqual(seed_summary([0., 0., 0.])['approximate_seed_t95'], [0., 0.])
        effect = seed_summary([.01, 0., .04])
        self.assertLess(effect['approximate_seed_t95'][0], 0.)
        self.assertEqual(effect['direction'], 'INCLUDES_ZERO')
        with self.assertRaises(ValueError):
            seed_summary([.01, .02])


if __name__ == '__main__':
    unittest.main()
