"""U2 intervention tests (written before the implementation). Test-time interventions on the immutable Core."""
import unittest

import numpy as np
import torch


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        try:
            from horizon import interventions as iv
        except ImportError as e:
            self.fail('U2 interventions module is not implemented: ' + str(e))
        return iv

    def make(self):
        from core.model import Core
        torch.manual_seed(11)
        return Core(width=8, heads=1, window=4, classes=5)

    def test_none_matches_forward_and_freeze_reverts_state_on_neutral(self):
        iv = self.api(); core = self.make()
        x = torch.tensor([[0, 1, 2, 2, 2, 1, 0, 2]])
        logits, states = iv.run(core, x, arm='none')
        torch.testing.assert_close(logits, core(x), rtol=1e-5, atol=1e-6)
        logits_f, states_f = iv.run(core, x, arm='freeze')
        # positions before the first neutral are identical; during the neutral run the state stays at the pre-run value
        torch.testing.assert_close(states_f[:, :2], states[:, :2])
        for t in (2, 3, 4):
            torch.testing.assert_close(states_f[:, t], states_f[:, 1])
            torch.testing.assert_close(logits_f[:, t], logits_f[:, 1])
        # after the run the frozen trajectory continues from the pre-run state (differs from the drifted one in general)
        self.assertEqual(tuple(states_f.shape), tuple(states.shape))

    def test_snap_moves_neutral_states_onto_nearest_anchor_and_wrong_snap_onto_another(self):
        iv = self.api(); core = self.make()
        x = torch.tensor([[0, 1, 2, 2, 1]])
        anchors = np.stack([np.full(8, i, dtype=np.float32) * 3 for i in range(5)])  # five distinct centers
        logits, states = iv.run(core, x, arm='snap', anchors=anchors)
        for t in (2, 3):
            d = np.linalg.norm(anchors - states[0, t].numpy(), axis=1)
            self.assertAlmostEqual(float(d.min()), 0., places=5)
        _, states_w = iv.run(core, x, arm='wrong_snap', anchors=anchors)
        for t in (2, 3):
            good = int(np.linalg.norm(anchors - states[0, t].numpy(), axis=1).argmin())
            wrong = int(np.linalg.norm(anchors - states_w[0, t].numpy(), axis=1).argmin())
            self.assertNotEqual(good, wrong)
        # non-neutral positions are untouched by snapping
        base_logits, _ = iv.run(core, x, arm='none')
        torch.testing.assert_close(logits[:, :2], base_logits[:, :2])

    def test_neutral_positions_are_detected_from_input_only(self):
        iv = self.api()
        x = torch.tensor([[2, 0, 2, 3]])
        self.assertEqual(iv.neutral_mask(x).tolist(), [[True, False, True, False]])


if __name__ == '__main__':
    unittest.main()
