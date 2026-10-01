"""U2b tests (written before the implementation): run-aware freeze — revert the state only from the (k+1)-th consecutive neutral."""
import unittest

import torch


class RunAwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import interventions_runaware as ivr
        except ImportError as e:
            self.fail('U2b run-aware module is not implemented: ' + str(e))
        return ivr

    def make(self):
        from core.model import Core
        torch.manual_seed(11)
        return Core(width=8, heads=1, window=4, classes=5)

    def test_run_length_counts_consecutive_neutrals_from_input_only(self):
        ivr = self.api()
        x = torch.tensor([[2, 2, 0, 2, 2, 2, 1], [0, 0, 0, 2, 0, 2, 2]])
        self.assertEqual(ivr.run_length(x).tolist(), [[1, 2, 0, 1, 2, 3, 0], [0, 0, 0, 1, 0, 1, 2]])

    def test_k2_leaves_short_runs_alone_and_freezes_from_the_third_neutral(self):
        ivr = self.api(); core = self.make()
        from horizon import interventions as iv
        x = torch.tensor([[0, 1, 2, 2, 2, 2, 1, 0]])
        base_logits, base_states = iv.run(core, x, arm='none')
        logits, states = ivr.run(core, x, k=2)
        # first two neutrals (t=2,3) are untouched
        torch.testing.assert_close(states[:, :4], base_states[:, :4])
        torch.testing.assert_close(logits[:, :4], base_logits[:, :4])
        # third and fourth neutrals are frozen at the second neutral's state and logits
        for t in (4, 5):
            torch.testing.assert_close(states[:, t], states[:, 3])
            torch.testing.assert_close(logits[:, t], logits[:, 3])
        self.assertEqual(tuple(states.shape), tuple(base_states.shape))

    def test_k0_equals_plain_freeze(self):
        ivr = self.api(); core = self.make()
        from horizon import interventions as iv
        x = torch.tensor([[0, 1, 2, 2, 2, 1, 0, 2]])
        logits_f, states_f = iv.run(core, x, arm='freeze')
        logits, states = ivr.run(core, x, k=0)
        torch.testing.assert_close(states, states_f)
        torch.testing.assert_close(logits, logits_f)


if __name__ == '__main__':
    unittest.main()
