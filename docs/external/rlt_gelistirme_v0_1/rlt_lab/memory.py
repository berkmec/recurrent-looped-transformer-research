"""Append-only raw archive + versioned heading metadata, with no fake embeddings.

SQLite triggers enforce append-only access through this schema; they are not
WORM storage and cannot resist an owner deleting the database. Backups are
still required. Vector retrieval is deliberately left as an integration API:
raw bytes are kept here, never reconstructed from an embedding or a summary.
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class StaleRevision(RuntimeError):
    pass


class Archive:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS blobs (
            digest TEXT PRIMARY KEY, raw BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS events (
            seq INTEGER PRIMARY KEY, heading TEXT NOT NULL, version INTEGER NOT NULL,
            model TEXT NOT NULL, digest TEXT NOT NULL, payload BLOB NOT NULL,
            previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL UNIQUE,
            UNIQUE(heading,version));
          CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
            BEGIN SELECT RAISE(ABORT,'append only'); END;
          CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
            BEGIN SELECT RAISE(ABORT,'append only'); END;
          CREATE TRIGGER IF NOT EXISTS blobs_no_update BEFORE UPDATE ON blobs
            BEGIN SELECT RAISE(ABORT,'append only'); END;
          CREATE TRIGGER IF NOT EXISTS blobs_no_delete BEFORE DELETE ON blobs
            BEGIN SELECT RAISE(ABORT,'append only'); END;
        ''')

    def close(self) -> None:
        self.db.close()

    def version(self, heading: str) -> int:
        return self.db.execute('SELECT COALESCE(MAX(version),0) FROM events '
                               'WHERE heading=?', (heading,)).fetchone()[0]

    def append(self, heading: str, raw: bytes, summary: str, *, model: str,
               expected_version: int, dependencies: dict[str, int] | None = None,
               evidence_status: str = 'CANDIDATE') -> dict[str, Any]:
        """Persist raw and summary atomically. Never overwrite prior versions.

        A dependency must reference an existing exact version. Existing stale
        dependencies are permitted and recorded; a controller must NOT reuse
        their derived results without freshness validation.
        """
        if not heading or not model or not isinstance(raw, bytes):
            raise ValueError('heading, model fingerprint and raw bytes required')
        if evidence_status not in {'CANDIDATE','VERIFIED','REJECTED','CHALLENGED'}:
            raise ValueError('unknown evidence_status')
        deps = dict(dependencies or {})
        self.db.execute('BEGIN IMMEDIATE')
        try:
            current = self.version(heading)
            if current != expected_version:
                raise StaleRevision(f'{heading}: expected {expected_version}, found {current}')
            for name, ver in deps.items():
                if not isinstance(ver, int) or ver < 1 or not self.db.execute(
                    'SELECT 1 FROM events WHERE heading=? AND version=?', (name,ver)
                ).fetchone():
                    raise ValueError(f'unknown dependency {name}@{ver}')
            digest = sha256(raw)
            self.db.execute('INSERT OR IGNORE INTO blobs VALUES (?,?)', (digest, raw))
            prev = self.db.execute('SELECT seq,event_hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            seq, parent = (1,'0'*64) if prev is None else (prev[0]+1, prev[1])
            item = dict(seq=seq, heading=heading, version=current+1, model=model,
                        digest=digest, summary=summary, dependencies=deps,
                        evidence_status=evidence_status)
            payload = canonical(item)
            event_hash = sha256(bytes.fromhex(parent)+payload)
            self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?)',
                            (seq,heading,current+1,model,digest,payload,parent,event_hash))
            self.db.execute('COMMIT')
            return {**item, 'event_hash':event_hash}
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def raw(self, digest: str) -> bytes:
        row = self.db.execute('SELECT raw FROM blobs WHERE digest=?',(digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        result = bytes(row[0])
        if sha256(result) != digest:
            raise ValueError('raw content hash mismatch')
        return result

    def verify(self) -> int:
        parent = '0'*64
        count = 0
        for seq,heading,version,model,digest,payload,previous,event_hash in self.db.execute(
                'SELECT * FROM events ORDER BY seq'):
            item = json.loads(payload)
            if seq != count+1 or previous != parent:
                raise ValueError('broken event sequence')
            if sha256(bytes.fromhex(previous)+payload) != event_hash:
                raise ValueError('broken hash chain')
            if (item['seq'],item['heading'],item['version'],item['model'],item['digest']) != (
                    seq,heading,version,model,digest):
                raise ValueError('metadata mismatch')
            self.raw(digest)
            parent = event_hash
            count += 1
        return count

    def backup(self, destination: str | Path) -> None:
        """Consistent SQLite backup incl. committed WAL data; then verify it.

        Caller chooses a different physical device for disk-loss protection.
        This function does not configure a Git remote or upload user data.
        """
        destination = Path(destination)
        if destination.resolve() == self.path.resolve():
            raise ValueError('backup destination must differ')
        destination.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(destination) as target:
            self.db.backup(target)
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
    archived: bool = False

    def __post_init__(self):
        if not self.heading or self.page < 0 or self.capacity < 1 or not 0 <= self.used <= self.capacity:
            raise ValueError('invalid heading page')

    @property
    def should_compact(self) -> bool:
        return self.used == self.capacity

    def consume(self, tokens: int) -> 'HeadingPage':
        if tokens < 0 or self.used + tokens > self.capacity:
            raise ValueError('invalid consumption; allocate next page at boundary')
        return HeadingPage(self.heading,self.page,self.capacity,self.used+tokens,False)

    def next_page(self, *, raw_persisted: bool, compacted: bool) -> 'HeadingPage':
        if not self.should_compact or not raw_persisted or not compacted:
            raise ValueError('boundary + persisted raw + compaction are required')
        return HeadingPage(self.heading,self.page+1,self.capacity)


def affected_descendants(changed: set[str], dependencies: dict[str,set[str]]) -> set[str]:
    """Return changed nodes AND their reverse dependency closure, cycles safe."""
    reverse: dict[str,set[str]] = {}
    for node,deps in dependencies.items():
        for dep in deps:
            reverse.setdefault(dep,set()).add(node)
    seen, queue = set(changed), list(changed)
    while queue:
        for child in reverse.get(queue.pop(),set()):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


def reusable(saved_dependencies: dict[str,int], current_versions: dict[str,int],
             *, saved_model: str, current_model: str) -> bool:
    """Freshness only, not a proof of semantic correctness."""
    return saved_model == current_model and all(
        current_versions.get(k) == v for k,v in saved_dependencies.items())
