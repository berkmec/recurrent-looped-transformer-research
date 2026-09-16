"""External integrity guard for the already-frozen combined runner."""
from pathlib import Path
import hashlib,json
from .safe_run import assert_protocol_integrity


def main():
    root=Path(__file__).resolve().parent
    primary=assert_protocol_integrity(root)
    path=root/'combined-protocol.json';actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=(root/'combined-protocol.sha256').read_text().strip():raise ValueError('combined protocol drift')
    protocol=json.loads(path.read_text())
    if primary!=protocol['primary_protocol_sha256']:raise ValueError('combined primary protocol pin mismatch')
    if hashlib.sha256((root/'combined_run.py').read_bytes()).hexdigest()!=protocol['source_sha256']:raise ValueError('combined source drift')
    from .combined_run import main as run_combined
    from .run import log
    log('combined_external_guard_verified',primary_protocol_sha256=primary,combined_protocol_sha256=actual,
        guard_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    run_combined()

if __name__=='__main__':main()
