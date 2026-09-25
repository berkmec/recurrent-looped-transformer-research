"""Nothing produced on the VM may exist only on the VM.

A rented box is not a durable asset: a preemption, a quota change, a billing lapse or one mistyped delete ends it
without warning. This campaign has already lost a local run to a power cut and had another killed by memory
pressure, and both times the cost was hours rather than evidence only because the evidence happened to be on the
machine that survived.

So the rule is mechanical, not a habit: a mirror process pulls the VM's results onto this machine continuously,
and pulls the checkpoints the moment a run's own record says it finished.

These tests pin the four properties that make it a guarantee rather than a convenience:

  the measurements come down on a SHORT clock, separately from the checkpoints, because losing a checkpoint costs
  compute while losing probe.jsonl costs the experiment;
  every checkpoint is VERIFIED by hash after transfer, since an unchecked copy is a copy you are guessing about;
  a finished run triggers its own pull immediately rather than waiting for the next slow cycle;
  and the mirror never writes into the repository's own results/, which holds sealed evidence from earlier cycles.
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PULLER = ROOT / 'scripts/pull_from_vm.py'


class MirrorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = PULLER.read_text(encoding='utf8')

    def api(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('pull_from_vm', PULLER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_measurements_and_checkpoints_are_separate_tiers(self):
        module = self.api()
        self.assertTrue(hasattr(module, 'light_pull'))
        self.assertTrue(hasattr(module, 'heavy_pull'))
        self.assertIn("--exclude='*.pt'", self.text, 'the frequent pull must not drag 125 MB checkpoints with it')

    def test_every_checkpoint_is_verified_after_transfer(self):
        self.assertIn('sha256 mismatch after transfer', self.text)
        self.assertIn('sha_remote', self.text)
        self.assertIn('sha_local', self.text)

    def test_a_finished_run_triggers_its_own_pull(self):
        self.assertIn('fresh or due', self.text)
        self.assertIn('pulling its checkpoints now', self.text)

    def test_completion_is_read_from_the_run_s_own_record(self):
        """Not from a guess about elapsed time: a run is finished when its log says so."""
        module = self.api()
        self.assertTrue(hasattr(module, 'completed_runs'))
        self.assertIn('complete.json', self.text)
        self.assertIn("len(rows) >= steps", self.text)

    def test_the_mirror_never_targets_the_repository_results(self):
        module = self.api()
        default = Path(module.DEFAULT_MIRROR).resolve()
        self.assertNotEqual(default, (ROOT / 'results').resolve())
        self.assertFalse(str(default).startswith(str(ROOT.resolve())),
                         'the mirror must live outside the repo, or a sync could overwrite sealed evidence')

    def test_a_dead_vm_does_not_kill_the_mirror_process(self):
        """The failure this exists for is the VM disappearing; the puller must survive it and say so."""
        self.assertIn('except Exception as exc', self.text)
        self.assertIn('pull failed', self.text)

    def test_completed_runs_reads_a_finished_probe(self):
        import json
        from tempfile import TemporaryDirectory
        module = self.api()
        with TemporaryDirectory() as directory:
            mirror = Path(directory)
            run = mirror / 'results' / 'M0-packed'
            run.mkdir(parents=True)
            (run / 'probe.json').write_text(json.dumps(dict(steps=3)), encoding='utf8')
            (run / 'probe.jsonl').write_text('{"step": 1}\n{"step": 2}\n', encoding='utf8')
            self.assertEqual(module.completed_runs(mirror), [], 'two of three steps is not finished')
            (run / 'probe.jsonl').write_text('{"step": 1}\n{"step": 2}\n{"step": 3}\n', encoding='utf8')
            self.assertEqual(module.completed_runs(mirror), ['M0-packed'])


if __name__ == '__main__':
    unittest.main()
