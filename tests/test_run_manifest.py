"""RUNMAN: run manifest v1 — atomic tmp+replace write, single-write refusal, dirty classification + full porcelain list,
schema validation, one-line quick() hook.

Written before the module binds into the run scripts; the schema is the approved v1 (_hive_KOSU_HIJYENI_SEMA.md, 7060c0).
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from horizon import run_manifest as rm


def make_spec(run_id='r1', campaign='Z1-pilot'):
    return dict(run_id=run_id, campaign=campaign, agent='jim-mv8403zi',
                command='bash scripts/run_faz1_jim.sh (a3-r1)', prereg={'paper': 'notes/_hive_A3_ONKAYIT.md'},
                config=dict(source='scripts/run_faz1_jim.sh', summary={'steps': 6000}),
                seeds={'init': 1, 'data_split': 1}, telemetry_schema='I1-a/v1', created_utc='2026-10-16T20:00:00Z')


CLEAN = dict(head='a31c52d', dirty=False, porcelain=[])


class RunManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_atomic_valid_and_single_write(self):
        with mock.patch.object(rm, 'git_state', return_value=dict(CLEAN)):
            path = rm.write_run_manifest(make_spec(), results_root=self.root)
        self.assertEqual(path, self.root / 'Z1-pilot' / 'r1.manifest.json')
        self.assertTrue(path.exists())
        self.assertFalse(path.with_name(path.name + '.tmp').exists())
        self.assertTrue(rm.validate(path))
        doc = json.loads(path.read_text(encoding='utf8'))
        self.assertEqual(doc['schema'], 'run-manifest/v1')
        self.assertEqual(doc['telemetry_schema'], 'I1-a/v1')
        self.assertEqual(doc['status'], 'started')
        with mock.patch.object(rm, 'git_state', return_value=dict(CLEAN)):
            with self.assertRaises(FileExistsError):
                rm.write_run_manifest(make_spec(), results_root=self.root)

    def test_failed_replace_leaves_no_half_file(self):
        with mock.patch.object(rm, 'git_state', return_value=dict(CLEAN)), \
             mock.patch.object(rm.os, 'replace', side_effect=OSError('boom')):
            with self.assertRaises(OSError):
                rm.write_run_manifest(make_spec('r2'), results_root=self.root)
        self.assertFalse((self.root / 'Z1-pilot' / 'r2.manifest.json').exists())
        self.assertEqual(list((self.root / 'Z1-pilot').glob('*')), [])

    def test_dirty_classification_and_full_list(self):
        dirty, lines = rm.classify_porcelain([' M horizon/depth_run.py', '?? results/Z1-pilot/x.json', '?? notes/y.md'])
        self.assertTrue(dirty)                                  # tracked code modified -> unsealed code
        self.assertEqual(len(lines), 3)                         # the complete list is kept either way
        dirty, _ = rm.classify_porcelain([' M results/M1-finetune-distance/queue.log', '?? results/Z1-pilot/x.json', '?? notes/y.md'])
        self.assertFalse(dirty)                                 # run outputs / notes move by design
        dirty, _ = rm.classify_porcelain(['?? horizon/gate_telemetry.py'])
        self.assertTrue(dirty)                                  # untracked code is unsealed code too
        dirty, _ = rm.classify_porcelain(['R  scripts/old.sh -> scripts/new.sh'])
        self.assertTrue(dirty)

    def test_validate_flags_inconsistency(self):
        dirty_state = dict(head='abc1234', dirty=True, porcelain=[' M horizon/depth_run.py'])
        with mock.patch.object(rm, 'git_state', return_value=dirty_state):
            path = rm.write_run_manifest(make_spec('r3'), results_root=self.root)
        self.assertTrue(rm.validate(path))
        doc = json.loads(path.read_text(encoding='utf8'))
        doc['code']['dirty'] = False                            # tampered: flag says clean, list shows a tracked change
        path.write_text(json.dumps(doc), encoding='utf8')
        self.assertFalse(rm.validate(path))

    def test_quick_hook(self):
        with mock.patch.object(rm, 'git_state', return_value=dict(CLEAN)):
            note = rm.quick('q1', 'T', 'jim-mv8403zi', 'echo hi', 'notes/x.md', '{"init": 7}',
                            'horizon/run_manifest.py', results_root=str(self.root))
        self.assertTrue(str(note).endswith('q1.manifest.json'))
        path = Path(note)
        self.assertTrue(rm.validate(path))
        doc = json.loads(path.read_text(encoding='utf8'))
        self.assertEqual(doc['seeds'], {'init': 7})
        self.assertEqual(doc['config']['source'], 'horizon/run_manifest.py')
        self.assertIsNotNone(doc['config']['sha256'])
        with mock.patch.object(rm, 'git_state', return_value=dict(CLEAN)):
            again = rm.quick('q1', 'T', 'jim-mv8403zi', 'echo hi', 'notes/x.md', '{}',
                             'horizon/run_manifest.py', results_root=str(self.root))
        self.assertIn('manifest exists', again)


if __name__ == '__main__':
    unittest.main()
