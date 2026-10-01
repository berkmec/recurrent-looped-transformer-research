import unittest,importlib.util,tempfile
import json,hashlib
from pathlib import Path


class PreserveEvidence(unittest.TestCase):
    def test_restart_refuses_existing_evidence_without_changes(self):
        self.assertIsNotNone(importlib.util.find_spec('learning.safe_run'),'fail-closed entrypoint absent')
        from learning.safe_run import assert_fresh_output
        owner=Path(__file__).resolve().parents[1]/'learning'
        with tempfile.TemporaryDirectory(dir=owner) as temp:
            root=Path(temp);(root/'results').mkdir();(root/'checkpoints').mkdir()
            assert_fresh_output(root)
            evidence=root/'results'/'prediction.npz';evidence.write_bytes(b'immutable-test-evidence')
            with self.assertRaises(FileExistsError):assert_fresh_output(root)
            self.assertEqual(evidence.read_bytes(),b'immutable-test-evidence')

    def test_protocol_drift_fails_even_when_output_folder_is_empty(self):
        import learning.safe_run as guard
        self.assertTrue(hasattr(guard,'assert_protocol_integrity'),'protocol integrity guard absent')
        owner=Path(__file__).resolve().parents[1]/'learning'
        with tempfile.TemporaryDirectory(dir=owner) as tmp:
            root=Path(tmp);path=root/'protocol.json';path.write_text('{"version":3}')
            digest=hashlib.sha256(path.read_bytes()).hexdigest();(root/'protocol.sha256').write_text(digest)
            source=root/'model.py';source.write_text('original model')
            sources={'model.py':hashlib.sha256(source.read_bytes()).hexdigest()}
            (root/'source-hashes.json').write_text(json.dumps(sources))
            (root/'execution.jsonl').write_text(json.dumps({'event':'protocol_frozen','sha256':digest,'sources':sources})+'\n')
            guard.assert_protocol_integrity(root)
            path.write_text('{"version":4}')
            with self.assertRaises(ValueError):guard.assert_protocol_integrity(root)
            path.write_text('{"version":3}');source.write_text('tampered model')
            (root/'source-hashes.json').write_text(json.dumps({'model.py':hashlib.sha256(source.read_bytes()).hexdigest()}))
            with self.assertRaises(ValueError):guard.assert_protocol_integrity(root)

if __name__=='__main__':unittest.main()
