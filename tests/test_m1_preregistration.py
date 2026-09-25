"""M1's pre-registration must live inside the frozen manifest, not in a ledger I can edit.

The campaign's own record is what forces this. For U4-sep2 I said repeatedly that "the decision rule was written in
code before the data arrived", and the timestamps say otherwise: the scoring module was first committed at 17:11
while two condition results had already landed at 16:05 and 16:31. What was genuinely fixed in advance were the
numeric thresholds - and they were fixed in `.superpowers/sdd/phase2/progress.md`, which is gitignored, unversioned
and writable by me at any moment. A pre-registration that only I can see and only I can date is not one.

So for M1 three things must hold, and each is checked here rather than asserted in prose:

  1. the thresholds and the named predictions are INSIDE `protocol`, hence inside `protocol_hash`, hence inside
     `frozen.sha256` - changing any of them after the freeze makes `verify_frozen` refuse;
  2. `mline/analyze.py`, the module that scores the experiment, is in the hashed `SOURCES` list, so the sealed
     evidence pins the rule itself and not merely the numbers it reads;
  3. the registered thresholds are the SAME OBJECT the rule uses at scoring time, so the two cannot drift apart
     silently the way the 64-slot and 512-slot gap/drift measurements did in U4-sep2.
"""
import unittest


class PreregistrationIsInsideTheFrozenProtocolTests(unittest.TestCase):
    def api(self):
        from mline import analyze as ma
        from mline import train as mt
        return mt, ma

    def test_protocol_carries_a_preregistration_block(self):
        mt, _ = self.api()
        pre = mt.protocol()['preregistration']
        for field in ('thresholds', 'predictions', 'statuses', 'falsification', 'registered_before'):
            self.assertIn(field, pre)

    def test_registered_thresholds_are_the_rule_s_own_thresholds(self):
        """If these can differ, the manifest records a rule that is not the rule that runs."""
        mt, ma = self.api()
        self.assertEqual(mt.protocol()['preregistration']['thresholds'], dict(ma.THRESHOLDS))

    def test_registered_statuses_are_exhaustive(self):
        mt, ma = self.api()
        self.assertEqual(tuple(mt.protocol()['preregistration']['statuses']), tuple(ma.STATUSES))

    def test_every_status_the_rule_can_return_is_registered(self):
        """A branch that returns an unregistered status is an outcome nobody predicted."""
        mt, ma = self.api()
        registered = set(mt.protocol()['preregistration']['statuses'])

        def arm(bpb, far, near):
            return dict(bits_per_byte=bpb,
                        needle={f'{k}@{n}': far for k in ('repeat', 'text') for n in (512, 2048)}
                               | {f'{k}@{n}@near': near for k in ('repeat', 'text') for n in (512, 2048)})

        seen = {
            ma.verdict(dict(plain=arm(1.5, 0., 0.)))['status'],                                              # INCOMPLETE
            ma.verdict(dict(plain=arm(1.5, 0., 0.), hard_st=arm(1.5, 0., 0.), soft=arm(1.5, 0., 0.)))['status'],
            ma.verdict(dict(plain=arm(1.5, 0., .9), hard_st=arm(1.5, 0., .9), soft=arm(1.5, 0., .9)))['status'],
            ma.verdict(dict(plain=arm(1.5, .2, .9), hard_st=arm(1.5, .8, .9), soft=arm(1.5, .3, .9)))['status'],
            ma.verdict(dict(plain=arm(1.5, .2, .9), hard_st=arm(2.5, .8, .9), soft=arm(1.5, .3, .9)))['status'],
        }
        self.assertTrue(seen <= registered, f'unregistered outcomes: {sorted(seen - registered)}')
        self.assertGreaterEqual(len(seen), 5, 'the probe cases should exercise at least five distinct outcomes')

    def test_the_scoring_module_is_hashed_with_the_evidence(self):
        mt, _ = self.api()
        self.assertIn('mline/analyze.py', mt.SOURCES)
        self.assertIn('mline/analyze.py', mt.source_hashes())

    def test_preregistration_is_covered_by_the_protocol_hash(self):
        """Editing a registered threshold must change the hash the frozen manifest is sealed with."""
        import copy

        from horizon import sweep as sw
        mt, _ = self.api()
        p = mt.protocol()
        tampered = copy.deepcopy(p)
        tampered['preregistration']['thresholds']['needle_gain'] = .5
        self.assertNotEqual(sw.digest(p), sw.digest(tampered))


class DisclosureTests(unittest.TestCase):
    """B1: the report must say which numbers are frozen and which are not, rather than leaving the reader to guess."""

    def test_plan_names_the_frozen_and_unfrozen_thresholds(self):
        from pathlib import Path
        text = Path('notes/M_HATTI_PLANI.md').read_text(encoding='utf8')
        self.assertIn('preregistration', text)
        for token in ('bpb_regression', 'needle_gain', 'measurable_floor'):
            self.assertIn(token, text, f'{token} is enforced by the rule and must be disclosed in the plan')


if __name__ == '__main__':
    unittest.main()
