"""Append-only, content-addressed raw archive with a hash chain and versioned heading dependencies.

Contracts (from the external design package, own implementation):
- raw bytes are stored losslessly and never rewritten; summaries are metadata, never a replacement for raw;
- a heading's next version needs the expected current version (stale writers are rejected);
- dependencies reference exact existing versions; freshness (`reusable`) is checked before any derived result is reused;
- SQLite triggers refuse UPDATE/DELETE on events and blobs (application-level append-only; not WORM hardware);
- page/compaction: no compaction before the page is full, raw must be persisted first, the next page keeps the heading id.
"""
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

EVIDENCE = ('CANDIDATE', 'VERIFIED', 'REJECTED', 'CHALLENGED', 'STALE')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class StaleRevision(RuntimeError):
    pass


class Archive:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS blobs (digest TEXT PRIMARY KEY, raw BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, heading TEXT NOT NULL, version INTEGER NOT NULL, model TEXT NOT NULL,
            digest TEXT NOT NULL, payload BLOB NOT NULL, previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL UNIQUE, UNIQUE(heading, version));
          CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'append only'); END;
          CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'append only'); END;
          CREATE TRIGGER IF NOT EXISTS blobs_no_update BEFORE UPDATE ON blobs BEGIN SELECT RAISE(ABORT, 'append only'); END;
          CREATE TRIGGER IF NOT EXISTS blobs_no_delete BEFORE DELETE ON blobs BEGIN SELECT RAISE(ABORT, 'append only'); END;
        ''')

    def close(self):
        self.db.close()

    def version(self, heading):
        return self.db.execute('SELECT COALESCE(MAX(version), 0) FROM events WHERE heading=?', (heading,)).fetchone()[0]

    def append(self, heading, raw, summary, *, model, expected_version, dependencies=None, evidence_status='CANDIDATE'):
        if not heading or not model or not isinstance(raw, bytes):
            raise ValueError('heading, model fingerprint and raw bytes are required')
        if evidence_status not in EVIDENCE:
            raise ValueError('unknown evidence_status')
        deps = dict(dependencies or {})
        self.db.execute('BEGIN IMMEDIATE')
        try:
            current = self.version(heading)
            if current != expected_version:
                raise StaleRevision(f'{heading}: expected version {expected_version}, found {current}')
            for name, ver in deps.items():
                if type(ver) is not int or ver < 1 or not self.db.execute('SELECT 1 FROM events WHERE heading=? AND version=?', (name, ver)).fetchone():
                    raise ValueError(f'unknown dependency {name}@{ver}')
            digest = sha256(raw)
            self.db.execute('INSERT OR IGNORE INTO blobs VALUES (?, ?)', (digest, raw))
            prev = self.db.execute('SELECT seq, event_hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            seq, parent = (1, '0' * 64) if prev is None else (prev[0] + 1, prev[1])
            item = dict(seq=seq, heading=heading, version=current + 1, model=model, digest=digest, summary=summary, dependencies=deps, evidence_status=evidence_status)
            payload = canonical(item); event_hash = sha256(bytes.fromhex(parent) + payload)
            self.db.execute('INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (seq, heading, current + 1, model, digest, payload, parent, event_hash))
            self.db.execute('COMMIT')
            return {**item, 'event_hash': event_hash}
        except BaseException:
            self.db.execute('ROLLBACK'); raise

    def raw(self, digest):
        row = self.db.execute('SELECT raw FROM blobs WHERE digest=?', (digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        data = bytes(row[0])
        if sha256(data) != digest:
            raise ValueError('raw content hash mismatch')
        return data

    def events(self, heading=None):
        query = 'SELECT payload FROM events' + (' WHERE heading=?' if heading else '') + ' ORDER BY seq'
        return [json.loads(r[0]) for r in self.db.execute(query, (heading,) if heading else ())]

    def verify(self):
        parent = '0' * 64; count = 0
        for seq, heading, version, model, digest, payload, previous, event_hash in self.db.execute('SELECT * FROM events ORDER BY seq'):
            item = json.loads(payload)
            if seq != count + 1 or previous != parent or sha256(bytes.fromhex(previous) + payload) != event_hash:
                raise ValueError('broken event chain')
            if (item['seq'], item['heading'], item['version'], item['model'], item['digest']) != (seq, heading, version, model, digest):
                raise ValueError('metadata mismatch')
            self.raw(digest); parent = event_hash; count += 1
        return count

    def backup(self, destination):
        destination = Path(destination)
        if destination.resolve() == self.path.resolve():
            raise ValueError('backup destination must differ from the archive path')
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(destination)
        try:
            self.db.backup(target)
        finally:
            target.close()  # an open handle would keep the file locked on Windows
        copy = Archive(destination)
        try:
            copy.verify()
        finally:
            copy.close()


@dataclass(frozen=True)
class HeadingPage:
    heading: str
    page: int
    capacity: int
    used: int = 0

    def __post_init__(self):
        if not self.heading or self.page < 0 or self.capacity < 1 or not 0 <= self.used <= self.capacity:
            raise ValueError('invalid heading page')

    @property
    def should_compact(self):
        return self.used == self.capacity

    def consume(self, tokens):
        if tokens < 0 or self.used + tokens > self.capacity:
            raise ValueError('invalid consumption; open the next page at the boundary')
        return HeadingPage(self.heading, self.page, self.capacity, self.used + tokens)

    def next_page(self, *, raw_persisted, compacted):
        if not self.should_compact:
            raise ValueError('no early compaction: the page is not full')
        if not raw_persisted or not compacted:
            raise ValueError('raw must be persisted and the page compacted before the next page opens')
        return HeadingPage(self.heading, self.page + 1, self.capacity)


def affected_descendants(changed, dependencies):
    """Changed nodes plus their reverse-dependency closure (cycle safe)."""
    reverse = {}
    for node, deps in dependencies.items():
        for dep in deps:
            reverse.setdefault(dep, set()).add(node)
    seen, queue = set(changed), list(changed)
    while queue:
        for child in reverse.get(queue.pop(), set()):
            if child not in seen:
                seen.add(child); queue.append(child)
    return seen


def reusable(saved_dependencies, current_versions, *, saved_model, current_model):
    """Freshness only: same model fingerprint and every saved dependency still at the saved version."""
    return saved_model == current_model and all(current_versions.get(k) == v for k, v in saved_dependencies.items())
