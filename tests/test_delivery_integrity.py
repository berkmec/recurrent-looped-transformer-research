import json
import pytest
from audit.delivery_integrity import build_manifest, verify_manifest, IntegrityError
import audit.delivery_integrity as integrity
import os


@pytest.fixture
def delivery(tmp_path):
    (tmp_path/'audit').mkdir()
    (tmp_path/'audit/campaign-closure.json').write_text(json.dumps({'status':'COMPLETE_BOUNDED_RESEARCH'}))
    (tmp_path/'raw.npz').write_bytes(b'original raw evidence')
    (tmp_path/'report.md').write_text('Measured result')
    return tmp_path


def test_same_length_content_change_is_rejected(delivery):
    build_manifest(delivery)
    (delivery/'raw.npz').write_bytes(b'ORIGINAL raw evidence')
    with pytest.raises(IntegrityError,match='changed'):
        verify_manifest(delivery)


def test_missing_and_unlisted_evidence_are_rejected(delivery):
    build_manifest(delivery)
    (delivery/'extra.json').write_text('{}')
    with pytest.raises(IntegrityError,match='inventory'):
        verify_manifest(delivery)
    (delivery/'extra.json').unlink()
    (delivery/'raw.npz').unlink()
    with pytest.raises(IntegrityError,match='inventory'):
        verify_manifest(delivery)


def test_raw_artifacts_included_caches_excluded_and_no_overwrite(delivery):
    (delivery/'model.pt').write_bytes(b'weights')
    (delivery/'traces.sqlite').write_bytes(b'archive')
    (delivery/'__pycache__').mkdir()
    (delivery/'__pycache__/ignored.pyc').write_bytes(b'cache')
    build_manifest(delivery)
    result=verify_manifest(delivery)
    assert result['files_verified']==5
    with pytest.raises(FileExistsError):
        build_manifest(delivery)


def test_unfinished_campaign_cannot_be_sealed(delivery):
    (delivery/'audit/campaign-closure.json').write_text(json.dumps({'status':'RUNNING'}))
    with pytest.raises(IntegrityError,match='not complete'):
        build_manifest(delivery)


def test_change_to_earlier_file_during_later_hash_is_rejected(delivery,monkeypatch):
    original=integrity.digest
    def changed_while_hashing(path):
        if path.name=='report.md':
            old=(delivery/'raw.npz').stat()
            (delivery/'raw.npz').write_bytes(b'ORIGINAL raw evidence')
            os.utime(delivery/'raw.npz',ns=(old.st_atime_ns,old.st_mtime_ns+10**9))
        return original(path)
    monkeypatch.setattr(integrity,'digest',changed_while_hashing)
    with pytest.raises(IntegrityError,match='changed while sealing'):
        build_manifest(delivery)
    assert not (delivery/'audit/final-manifest.json').exists()
