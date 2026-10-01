"""B4/B6/D6/E6: append-only content-addressed archive with a hash chain, versioned heading dependencies, page/compaction contract."""
import sqlite3
import tempfile
import unittest
from pathlib import Path


class ArchiveTests(unittest.TestCase):
    def api(self):
        try:
            from archive import store
        except ImportError as e:
            self.fail('archive.store is not implemented: ' + str(e))
        return store

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_raw_bytes_are_lossless_and_summary_never_replaces_raw(self):
        st = self.api(); a = st.Archive(self.root / 'a.sqlite')
        raw = 'Başlık İ/ı/ğ satır\r\nikinci satır\n'.encode('utf8')
        e1 = a.append('A', raw, 'kısa özet', model='m1', expected_version=0)
        self.assertEqual(a.raw(e1['digest']), raw)
        e2 = a.append('A', b'v2 bytes', 'özet 2', model='m1', expected_version=1)
        self.assertEqual(a.raw(e1['digest']), raw)  # old version still readable, byte-exact
        self.assertEqual(e2['version'], 2); self.assertEqual(a.version('A'), 2)
        a.close()

    def test_stale_writer_unknown_dependency_and_mutation_are_rejected(self):
        st = self.api(); a = st.Archive(self.root / 'a.sqlite')
        a.append('A', b'a1', 's', model='m1', expected_version=0)
        with self.assertRaises(st.StaleRevision):
            a.append('A', b'a2', 's', model='m1', expected_version=0)
        with self.assertRaises(ValueError):
            a.append('B', b'b1', 's', model='m1', expected_version=0, dependencies={'A': 2})
        a.append('B', b'b1', 's', model='m1', expected_version=0, dependencies={'A': 1})
        db = sqlite3.connect(self.root / 'a.sqlite')
        with self.assertRaises(sqlite3.DatabaseError):
            db.execute('DELETE FROM events')
        with self.assertRaises(sqlite3.DatabaseError):
            db.execute("UPDATE events SET heading='Z'")
        db.close(); a.close()

    def test_hash_chain_verify_and_backup(self):
        st = self.api(); a = st.Archive(self.root / 'a.sqlite')
        for i in range(8):
            a.append('A', f'v{i}'.encode(), 's', model='m1', expected_version=i)
        self.assertEqual(a.verify(), 8)
        a.backup(self.root / 'b.sqlite')
        b = st.Archive(self.root / 'b.sqlite'); self.assertEqual(b.verify(), 8); b.close()
        with self.assertRaises(ValueError):
            a.backup(self.root / 'a.sqlite')
        a.close()

    def test_dependency_closure_and_freshness(self):
        st = self.api()
        deps = {'b': {'a'}, 'c': {'b'}, 'd': {'x'}, 'a': {'c'}}  # cycle a->b->c->a, d independent
        self.assertEqual(st.affected_descendants({'a'}, deps), {'a', 'b', 'c'})
        self.assertTrue(st.reusable({'a': 1}, {'a': 1, 'b': 5}, saved_model='m1', current_model='m1'))
        self.assertFalse(st.reusable({'a': 1}, {'a': 2}, saved_model='m1', current_model='m1'))
        self.assertFalse(st.reusable({'a': 1}, {'a': 1}, saved_model='m1', current_model='m2'))

    def test_page_boundary_requires_archive_then_compaction_same_heading(self):
        st = self.api()
        page = st.HeadingPage('A', page=0, capacity=4)
        page = page.consume(3)
        with self.assertRaises(ValueError):
            page.next_page(raw_persisted=True, compacted=True)  # not full: no early compaction
        page = page.consume(1)
        self.assertTrue(page.should_compact)
        with self.assertRaises(ValueError):
            page.next_page(raw_persisted=False, compacted=True)
        nxt = page.next_page(raw_persisted=True, compacted=True)
        self.assertEqual((nxt.heading, nxt.page, nxt.used), ('A', 1, 0))


if __name__ == '__main__':
    unittest.main()
