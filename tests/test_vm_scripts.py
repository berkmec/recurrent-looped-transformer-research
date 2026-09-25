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


class CalibrationAndParallelTests(unittest.TestCase):
    """The 24 GB card re-opens batch size and concurrency; both are decided by measurement, in that order."""

    def test_the_run_script_calibrates_before_m1(self):
        text = RUN.read_text(encoding='utf8')
        self.assertLess(text.index('calibrate.py'), text.index('M1 freeze'))

    def test_the_plan_comes_from_the_measurement_not_from_a_guess(self):
        text = RUN.read_text(encoding='utf8')
        self.assertIn("calibration.json", text)
        self.assertIn('PLAN=', text)
        self.assertIn('if [ "$PLAN" = "parallel" ]', text)

    def test_parallel_workers_take_their_idents_from_the_protocol(self):
        """A condition name typed into a shell script drifts the moment the width or the seed changes."""
        from mline import train as mt
        text = RUN.read_text(encoding='utf8')
        self.assertIn('--only', text)
        self.assertIn('t.ident(c) for c in t.conditions(t.protocol())', text)
        for c in mt.conditions(mt.protocol()):
            self.assertNotIn(mt.ident(c), text, f'{mt.ident(c)} is hardcoded; derive it instead')

    def test_a_failed_worker_falls_back_to_a_serial_resume(self):
        text = RUN.read_text(encoding='utf8')
        self.assertIn('M1 run (recovery)', text)
        self.assertIn('resumes each arm from its own checkpoint', text)

    def test_calibration_compares_throughput_not_step_time(self):
        text = (ROOT / 'scripts/calibrate.py').read_text(encoding='utf8')
        self.assertIn('bytes_per_s', text)
        self.assertIn('never seconds per step', text)

    def test_calibration_does_not_silently_change_the_experiment(self):
        """Concurrency is operational; batch size is part of the protocol and changes what the arms see."""
        run = RUN.read_text(encoding='utf8')
        self.assertIn("['plan']", run, 'the run script must read only the plan')
        self.assertNotIn("['batch']", run, 'reading the calibrated batch would change the frozen experiment')
        self.assertIn('advisory only', (ROOT / 'scripts/calibrate.py').read_text(encoding='utf8'))

    def test_the_parallel_threshold_is_stated_rather_than_implied(self):
        text = (ROOT / 'scripts/calibrate.py').read_text(encoding='utf8')
        self.assertIn('1.5', text)
        self.assertIn('recorded here so it can', text, 'a judgement call must be visible, not buried in a branch')

    def test_the_parallel_batch_must_fit_the_workers(self):
        """Measured the hard way: three workers at the best single batch needed 42.7 GB on a 22.5 GB card."""
        text = (ROOT / 'scripts/calibrate.py').read_text(encoding='utf8')
        self.assertIn('total_vram_gb', text)
        self.assertIn('* len(arms) <= budget', text)
        self.assertIn('memory-fraction', text)

    def test_the_speedup_compares_the_same_batch(self):
        """Comparing a parallel aggregate against a different single batch mixes two effects into one verdict."""
        text = (ROOT / 'scripts/calibrate.py').read_text(encoding='utf8')
        self.assertIn('== par_batch', text)
        self.assertIn('never against the', text)

