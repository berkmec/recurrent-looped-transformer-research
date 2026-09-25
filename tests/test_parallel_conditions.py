"""M1 has exactly three arms and the rented card has 24 GB, so running them concurrently becomes possible.

The campaign's "GPU jobs serial" rule was adopted for an 8 GB card that could not hold two 30M arms at once. That
is a hardware fact, not a principle, and a card with three times the memory re-opens it. Whether concurrency
actually pays is unknown and must be measured: a compute-bound loop gains nothing from sharing one GPU, a
launch-bound one overlaps and turns M1's 22 hours into about 8. I have already been caught carrying a 5 per cent
utilisation figure over from the toy models, so the decision waits for the number.

What can be settled in advance is the mechanism. Running the arms in separate processes needs a way to ask for
ONE condition, and the existing safety properties have to survive it: a completed condition is skipped, a
condition already being worked on is locked, and no two processes may claim the same one.
"""
import unittest


class SingleConditionSelectionTests(unittest.TestCase):
    def mt(self):
        from mline import train as mt
        return mt

    def test_conditions_have_stable_unique_identifiers(self):
        mt = self.mt()
        p = mt.protocol()
        idents = [mt.ident(c) for c in mt.conditions(p)]
        self.assertEqual(len(idents), len(set(idents)), 'parallel workers select by ident; duplicates would collide')
        self.assertEqual(len(idents), len(p['arms']) * len(p['seeds']))

    def test_the_cli_can_run_one_condition(self):
        from pathlib import Path
        text = Path(__file__).resolve().parents[1].joinpath('mline/train.py').read_text(encoding='utf8')
        self.assertIn("'--only'", text, 'a parallel worker needs to ask for a single condition')

    def test_an_unknown_ident_is_refused_rather_than_silently_doing_nothing(self):
        from pathlib import Path
        text = Path(__file__).resolve().parents[1].joinpath('mline/train.py').read_text(encoding='utf8')
        self.assertIn('no condition named', text)

    def test_the_lock_and_completion_guards_are_still_in_place(self):
        """Concurrency is only safe because these already existed; a test so they are not removed later."""
        from pathlib import Path
        text = Path(__file__).resolve().parents[1].joinpath('mline/train.py').read_text(encoding='utf8')
        self.assertIn('active.lock', text)
        self.assertIn("complete.json", text)

    def test_every_arm_is_selectable_by_its_own_ident(self):
        mt = self.mt()
        p = mt.protocol()
        for c in mt.conditions(p):
            self.assertIn(c['arm'], mt.ident(c), 'the ident must name the arm, or a worker cannot be told what to run')


if __name__ == '__main__':
    unittest.main()
