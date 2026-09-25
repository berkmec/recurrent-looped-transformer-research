"""Name the binary assets this suite needs, and where they come from, so a missing one fails legibly.

Found on 2026-09-25, the first time the runner actually executed every test in the directory: seven tests in
`test_reasoning_window.py` and `test_integration_pipeline.py` were failing with a bare

    FileNotFoundError: E:\\rlt-research\\2026-09-18-phase2\\memory\\checkpoints\\extractor-17.pt

thrown from inside `torch.serialization`, six frames below anything this worktree wrote. They call
`memory.api.load_parser()` with no argument, which defaults to a phase-1 training checkpoint that exists only in
the sealed `2026-09-16-e2e` root. The file is `*.pt`, so `.gitignore` keeps it out of the repository and a fresh
checkout has never had it.

The tests themselves cannot carry the guard: they are hashed in `audit/final-manifest.json`,
`audit/drift-mini-manifest.json` and `audit/review-diagnostics-manifest.json`, and editing them would break three
seals to improve an error message. `memory/api.py` is hashed in the B1 and D1 frozen manifests for the same
reason. So the check lives here, in a new and unsealed file, and says the one thing the traceback did not: which
file is missing and where to copy it from.
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT.parent / '2026-09-16-e2e'

REQUIRED = {
    'memory/checkpoints/extractor-17.pt': dict(
        source=PHASE1 / 'memory' / 'checkpoints' / 'extractor-17.pt',
        needed_by='memory.api.load_parser() default; 7 tests in test_reasoning_window.py and '
                  'test_integration_pipeline.py call it with no argument',
        note='phase-1 training artefact, gitignored as *.pt; copy it, do not regenerate it, so the parser under '
             'test stays the one phase 1 measured'),
}


class RequiredAssetTests(unittest.TestCase):
    def test_every_required_asset_is_present_or_the_message_says_where_to_get_it(self):
        missing = []
        for relative, info in REQUIRED.items():
            if (ROOT / relative).exists():
                continue
            available = info['source'].exists()
            missing.append('%s is missing. Needed by: %s. %s Copy from: %s%s'
                           % (relative, info['needed_by'], info['note'], info['source'],
                              '' if available else '  (NOT PRESENT THERE EITHER - this asset is lost)'))
        self.assertEqual(missing, [], '\n'.join(missing))

    def test_the_recorded_source_is_a_read_only_reference(self):
        """Phase-1 roots are sealed: this suite may copy OUT of them and must never write INTO them."""
        for info in REQUIRED.values():
            self.assertIn('2026-09-16-e2e', str(info['source']))
            self.assertFalse(str(info['source']).startswith(str(ROOT) + '\\'),
                             'a required asset must not be sourced from inside the worktree that consumes it')


if __name__ == '__main__':
    unittest.main()
