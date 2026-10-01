import unittest,importlib.util,tempfile
from pathlib import Path


class CombinedProtocol(unittest.TestCase):
    def test_combined_restart_guard_preserves_primary_and_combined_evidence(self):
        self.assertIsNotNone(importlib.util.find_spec('learning.combined_run'),'predeclared combined runner absent')
        from learning.combined_run import assert_fresh_combined
        owner=Path(__file__).resolve().parents[1]/'learning'
        with tempfile.TemporaryDirectory(dir=owner) as tmp:
            root=Path(tmp);(root/'checkpoints').mkdir();(root/'results').mkdir()
            (root/'checkpoints'/'bidir-17-step600.pt').write_bytes(b'primary')
            assert_fresh_combined(root)
            path=root/'checkpoints'/'combined-17-initial.pt';path.write_bytes(b'combined')
            with self.assertRaises(FileExistsError):assert_fresh_combined(root)
            self.assertEqual(path.read_bytes(),b'combined')

if __name__=='__main__':unittest.main()
