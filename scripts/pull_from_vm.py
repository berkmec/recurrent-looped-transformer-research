"""Mirror the VM's results onto this machine continuously, because a rented box is not a durable asset.

THE RULE THIS ENFORCES: nothing produced on the VM is allowed to exist only on the VM. A preemption, a quota
change, a billing lapse or a mistyped `gcloud compute instances delete` ends that machine without warning, and
the campaign has already lost one local run to a power cut and had another killed by memory pressure. Evidence
that lives in one place is evidence that can be lost in one event.

Two tiers, because the sizes differ by four orders of magnitude:

  LIGHT   every cycle: every file except checkpoints - probe.jsonl, train.json, the logs, the manifests. Tens of
          kilobytes, so it can run often, and it carries all the MEASUREMENTS. Losing a checkpoint costs compute;
          losing the measurements costs the experiment.
  HEAVY   on completion, and on a slower clock: the *.pt files, about 125 MB each. Pulled when a run finishes,
          and otherwise only occasionally, because re-fetching a rolling checkpoint every few minutes would
          saturate a home uplink for no gain.

Every heavy file is VERIFIED after transfer by comparing sha256 against the remote. A copy that was not checked
is a copy you are guessing about, and this campaign has already had a checkpoint that wrote fine and could not be
read.

The mirror deliberately does NOT write into the repository's own `results/`. That directory holds sealed evidence
from earlier cycles, and a sync that overwrote it would destroy the thing it exists to protect.

Usage (locally, in Git Bash):
    python -B scripts/pull_from_vm.py --host USER@HOST --key ~/.ssh/KEYFILE --once
    python -B scripts/pull_from_vm.py --host ... --key ... --interval 600
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_REMOTE = '/data/rlt/2026-09-18-phase2'
DEFAULT_MIRROR = 'E:/rlt-research/vm-mirror'
# Only what the VM PRODUCES. Its results/ also holds the 225 MB of sealed evidence that was uploaded so the test
# suite could run there, and pulling that back would re-download what this machine is the source of. Measured:
# the first version of this puller stalled for ten minutes doing exactly that.
PRODUCED = ('results/M0-packed', 'results/M0-padded', 'results/M1', 'results/calibration.json')


def ssh(key, host, command, binary=False):
    out = subprocess.run(['ssh', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=25', host, command],
                         capture_output=True, text=not binary)
    return out


def light_pull(key, host, remote, mirror, paths=PRODUCED):
    """Everything except checkpoints, as one tar stream, scoped to what this VM produced."""
    mirror.mkdir(parents=True, exist_ok=True)
    wanted = ' '.join(paths) + ' results/vm-run-*.log'
    command = ("cd %s && tar czf - --exclude='*.pt' --exclude='__pycache__' $(ls -d %s 2>/dev/null) 2>/dev/null"
               % (remote, wanted))
    sender = subprocess.Popen(['ssh', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=25', host, command],
                              stdout=subprocess.PIPE)
    receiver = subprocess.Popen(['tar', 'xzf', '-', '-C', str(mirror)], stdin=sender.stdout)
    sender.stdout.close()
    receiver.communicate()
    sender.wait()
    return receiver.returncode == 0


def remote_checkpoints(key, host, remote):
    scope = ' '.join('%s/%s' % (remote, p) for p in PRODUCED if not p.endswith('.json'))
    # previous.pt is the VM's own insurance against a corrupt write THERE. Mirroring it doubles the traffic for
    # a copy that would never be used, because latest.pt is verified by sha256 as it lands here.
    out = ssh(key, host,
              "find %s -name '*.pt' ! -name 'previous.pt' -printf '%%s %%p\\n' 2>/dev/null" % scope)
    rows = {}
    for line in out.stdout.splitlines():
        size, _, path = line.partition(' ')
        if path:
            rows[path.strip()] = int(size)
    return rows


def sha_remote(key, host, path):
    out = ssh(key, host, f"sha256sum '{path}' 2>/dev/null | cut -d' ' -f1")
    return out.stdout.strip()


def sha_local(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def heavy_pull(key, host, remote, mirror, report=print):
    """Checkpoints, fetched only when missing or changed, and VERIFIED by hash after transfer."""
    pulled, verified, failed = [], [], []
    for path, size in remote_checkpoints(key, host, remote).items():
        relative = Path(path).relative_to(remote)
        target = mirror / relative
        want = sha_remote(key, host, path)
        if target.exists() and target.stat().st_size == size and sha_local(target) == want:
            continue                                     # already here and identical; nothing to do
        target.parent.mkdir(parents=True, exist_ok=True)
        report(f'  fetching {relative} ({size / 1e6:.0f} MB)')
        out = subprocess.run(['scp', '-i', key, '-o', 'BatchMode=yes', '-q', f'{host}:{path}', str(target)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            failed.append((str(relative), out.stderr.strip()[-160:])); continue
        pulled.append(str(relative))
        # A copy that was not checked is a copy you are guessing about.
        if want and sha_local(target) == want:
            verified.append(str(relative))
        else:
            failed.append((str(relative), 'sha256 mismatch after transfer'))
    return dict(pulled=pulled, verified=verified, failed=failed)


def completed_runs(mirror):
    """Runs whose own record says they finished, read from the mirror rather than asked of the VM."""
    done = []
    for header in sorted(mirror.glob('results/*/probe.json')):
        try:
            steps = json.loads(header.read_text(encoding='utf8'))['steps']
            rows = (header.parent / 'probe.jsonl').read_text(encoding='utf8').strip().splitlines()
        except Exception:
            continue
        if len(rows) >= steps:
            done.append(header.parent.name)
    for receipt in sorted(mirror.glob('results/*/*/complete.json')):
        done.append(f'{receipt.parent.parent.name}/{receipt.parent.name}')
    return done


def cycle(key, host, remote, mirror, state, heavy_every, report=print):
    ok = light_pull(key, host, remote, mirror)
    finished = set(completed_runs(mirror))
    fresh = finished - set(state.get('pulled_heavy', []))
    state['cycles'] = state.get('cycles', 0) + 1
    due = state['cycles'] % max(heavy_every, 1) == 0
    if fresh:
        report(f'run finished: {", ".join(sorted(fresh))} - pulling its checkpoints now')
    if fresh or due:
        result = heavy_pull(key, host, remote, mirror, report)
        if result['pulled'] or result['failed']:
            report(f"  checkpoints: {len(result['verified'])} verified, {len(result['failed'])} failed")
        for name in fresh:
            if not result['failed']:
                state.setdefault('pulled_heavy', []).append(name)
    return ok, sorted(finished)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--remote', default=DEFAULT_REMOTE)
    parser.add_argument('--mirror', default=DEFAULT_MIRROR)
    parser.add_argument('--interval', type=int, default=600, help='seconds between light pulls')
    parser.add_argument('--heavy-every', type=int, default=6,
                        help='also pull checkpoints every Nth cycle as insurance against losing the VM mid-run. '
                             'A checkpoint is 374 MB (weights plus AdamW moments), so this is the expensive '
                             'clock; a finished run pulls immediately regardless of it.')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()

    mirror = Path(args.mirror)
    state_path = mirror / 'pull-state.json'
    state = json.loads(state_path.read_text(encoding='utf8')) if state_path.exists() else {}

    while True:
        stamp = time.strftime('%H:%M:%S')
        try:
            ok, finished = cycle(args.key, args.host, args.remote, mirror, state, args.heavy_every)
            size = sum(f.stat().st_size for f in mirror.rglob('*') if f.is_file())
            print(f'[{stamp}] mirror {"ok" if ok else "PARTIAL"} | {size / 1e6:.1f} MB local | '
                  f'finished: {", ".join(finished) or "none yet"}', flush=True)
        except Exception as exc:                          # the VM may be gone; that is the case this exists for
            print(f'[{stamp}] pull failed: {type(exc).__name__}: {exc}', flush=True)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=1) + '\n', encoding='utf8')
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
