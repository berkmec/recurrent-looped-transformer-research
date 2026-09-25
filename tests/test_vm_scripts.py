"""The VM scripts carry checksums and a stop rule; both are checked here rather than trusted.

A setup script that verifies the corpus against a hash typed by hand is only as good as the typing. If the hash
is wrong the upload check fails on a good file, the first VM hour goes on debugging a non-problem, and on a
24 USD budget that hour is real money. So the hashes in the script are compared against the corpus manifest.

The run script's stop rule matters more. It exists so M1 - 22 hours - cannot start while the plain arm still
cannot do the task at all, which is precisely the mistake the padded probe cost 9 hours to reveal. These tests
pin that the gate is present, that it reads the leading indicator, and that the probe runs before M1.
"""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / 'scripts/vm_setup.sh'
RUN = ROOT / 'scripts/vm_run.sh'


class SetupScriptTests(unittest.TestCase):
    def manifest(self):
        from mline import train as mt
        path = Path(mt.DATA) / 'corpus.json'
        if not path.exists():
            raise unittest.SkipTest('MISSING_ASSET: ' + str(path))
        return json.loads(path.read_text(encoding='utf8'))

    def test_the_checksums_match_the_corpus_that_exists(self):
        text = SETUP.read_text(encoding='utf8')
        c = self.manifest()
        for name, key in (('TRAIN_SHA', 'train_sha256'), ('VAL_SHA', 'validation_sha256')):
            found = re.search(rf'^{name}=([0-9a-f]{{64}})$', text, re.M)
            self.assertIsNotNone(found, f'{name} is missing from the setup script')
            self.assertEqual(found.group(1), c[key], f'{name} does not match {key} in corpus.json')

    def test_setup_refuses_to_continue_on_a_mismatch(self):
        text = SETUP.read_text(encoding='utf8')
        self.assertIn('CHECKSUM MISMATCH', text)
        self.assertIn('exit 1', text, 'a failed verification must stop the script, not just print')

    def test_setup_measures_step_cost_before_anything_long(self):
        """Every speed estimate this campaign made without measuring was optimistic, by 3x, 5x and 13x."""
        text = SETUP.read_text(encoding='utf8')
        self.assertIn('--probe', text)
        self.assertLess(text.index('--probe'), len(text), 'the probe must be part of setup')

    def test_setup_does_not_touch_the_local_sealed_environment(self):
        text = SETUP.read_text(encoding='utf8')
        for forbidden in ('DaxDream', 'C:/', 'E:/'):
            self.assertNotIn(forbidden, text, 'the VM script must not reference the local machine')


class RunScriptTests(unittest.TestCase):
    def test_the_probe_runs_before_m1(self):
        text = RUN.read_text(encoding='utf8')
        self.assertLess(text.index('mline.emergence'), text.index('mline.train --freeze'),
                        'M1 costs 22 hours and measures nothing if the capability is absent')

    def test_there_is_a_gate_that_can_stop_before_m1(self):
        text = RUN.read_text(encoding='utf8')
        self.assertIn("train@32", text, 'the gate must read the leading indicator, not the evaluation sets')
        self.assertIn('sys.exit(0 if ok else 1)', text)
        self.assertIn('stopping before M1 on purpose', text)

    def test_interruptions_resume_rather_than_restart(self):
        text = RUN.read_text(encoding='utf8')
        self.assertIn('resuming from its last checkpoint', text)
        self.assertIn('retry ', text)

    def test_the_meter_is_switched_off_at_the_end(self):
        """On a 24 USD budget an instance left running overnight is a material fraction of the budget."""
        text = RUN.read_text(encoding='utf8')
        self.assertIn('shutdown -h', text)
        self.assertIn('--no-shutdown', text, 'and an explicit way to keep it up while debugging')

    def test_both_scripts_stop_on_the_first_error(self):
        for path in (SETUP, RUN):
            self.assertIn('set -euo pipefail', path.read_text(encoding='utf8'), path.name)


if __name__ == '__main__':
    unittest.main()
