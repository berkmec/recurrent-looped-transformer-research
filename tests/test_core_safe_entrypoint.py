import hashlib,json,tempfile,unittest
from pathlib import Path
from core.safe_entrypoint import check_entry,Refusal,verify_file_receipts,verify_manifest_only,bind_completed
class SafeEntry(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'core');self.r=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_partial_refused_without_writes(self):
  (self.r/'checkpoints').mkdir();p=self.r/'checkpoints'/'partial.pt';p.write_bytes(b'partial');before=p.read_bytes()
  with self.assertRaises(Refusal):check_entry(self.r)
  self.assertEqual(before,p.read_bytes())
 def test_binding_drift_refused(self):
  (self.r/'results').mkdir();p=self.r/'results'/'metrics.json';p.write_text('{"status":"COMPLETE"}');f=self.r/'results'/'final-only-metrics.json';f.write_text('{"status":"COMPLETE"}')
  for name in ['protocol.json','protocol.sha256','environment.json','model.py','run.py']:(self.r/name).write_text('fixture')
  b={'files':{x.relative_to(self.r).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in self.r.rglob('*') if x.is_file()}};(self.r/'evidence-binding.json').write_text(json.dumps(b));verify_manifest_only(self.r,b)
  with self.assertRaises(Refusal):check_entry(self.r)
  p.write_text('changed')
  with self.assertRaises(Refusal):verify_manifest_only(self.r,b)
 def test_prebinding_npz_and_checkpoint_drift(self):
  for name in ['predictions.npz','selected.pt']:
   p=self.r/name;p.write_bytes(b'original');receipts={name:hashlib.sha256(p.read_bytes()).hexdigest()};verify_file_receipts(self.r,receipts);p.write_bytes(b'drift')
   with self.assertRaises(Refusal):verify_file_receipts(self.r,receipts)
 def test_no_evidence_is_new(self):self.assertEqual(check_entry(self.r),'NEW')
 def test_binding_not_created_for_fake_complete_status(self):
  (self.r/'results').mkdir();(self.r/'results'/'metrics.json').write_text('{"status":"COMPLETE","training":[],"evaluation":[]}');(self.r/'results'/'final-only-metrics.json').write_text('{"status":"COMPLETE","records":[]}')
  with self.assertRaises(Refusal):bind_completed(self.r)
  self.assertFalse((self.r/'evidence-binding.json').exists())
if __name__=='__main__':unittest.main()
