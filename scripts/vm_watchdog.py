"""Watch the VM, and the moment its work ends, secure the data and switch the meter off.

The requirement is simple and the failure it prevents is expensive: a rented box that has finished its work and
is still billing. Two things stood between us and that. The pipeline's own shutdown carries a 240-minute grace
so results can be collected, and my monitor checked every 30 minutes - together, up to four and a half hours of
an idle machine drawing power for nothing.

This removes both, and removes my attention from the critical path. It polls on a short clock; when the pipeline
is gone it pulls everything down, VERIFIES it, and only then powers the box off. The order is not negotiable: a
machine shut down before its data is verified here is a machine that took the data with it.

It also refuses to shut down on a stall. A pipeline whose process died mid-run is not a pipeline that finished,
and the difference is visible in whether the run's own record reached its step count.

Usage (locally):
    python -B scripts/vm_watchdog.py --host USER@HOST --key ~/.ssh/key --interval 600
    python -B scripts/vm_watchdog.py --host ... --key ... --no-shutdown    # watch and report only
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ssh(key, host, command, timeout=60):
    return subprocess.run(['ssh', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=25', host, command],
                          capture_output=True, text=True, timeout=timeout)


def pipeline_state(key, host, remote):
    """What the VM says about itself: is the runner alive, and how far did each run get?"""
    probe = (f"pgrep -f vm_run.sh >/dev/null && echo alive || echo gone; "
             f"for d in {remote}/results/M0-packed {remote}/results/M0-padded; do "
             f"  n=$(wc -l < $d/probe.jsonl 2>/dev/null || echo 0); "
             f"  s=$(python3 -c \"import json,sys;print(json.load(open('$d/probe.json'))['steps'])\" 2>/dev/null || echo 0); "
             f"  echo \"$(basename $d) $n $s\"; done; "
             f"echo m1 $(ls {remote}/results/M1/*/complete.json 2>/dev/null | wc -l)")
    out = ssh(key, host, probe)
    if out.returncode != 0:
        return dict(reachable=False, raw=out.stderr.strip()[-200:])
    lines = out.stdout.strip().splitlines()
    state = dict(reachable=True, alive=lines[0].strip() == 'alive', runs={}, m1_done=0)
    for line in lines[1:]:
        parts = line.split()
        if parts and parts[0] == 'm1':
            state['m1_done'] = int(parts[1])
        elif len(parts) == 3:
            state['runs'][parts[0]] = dict(steps=int(parts[1]), total=int(parts[2]))
    return state


def secure_and_verify(key, host, remote, mirror, report):
    """Pull everything, then check the mirror against the VM. Returns True only if every file matched."""
    from scripts import pull_from_vm as pull                       # same code path the continuous mirror uses
    ok = pull.light_pull(key, host, remote, mirror)
    heavy = pull.heavy_pull(key, host, remote, mirror, report)
    if not ok or heavy['failed']:
        report(f"  transfer incomplete: light={ok} failed={heavy['failed']}")
        return False
    # Independent re-check rather than trusting the transfer report, on the files that matter most.
    missing = []
    for name in ('M0-packed', 'M0-padded'):
        header = mirror / 'results' / name / 'probe.json'
        rows = mirror / 'results' / name / 'probe.jsonl'
        if not header.exists():
            continue
        want = json.loads(header.read_text(encoding='utf8'))['steps']
        got = len(rows.read_text(encoding='utf8').strip().splitlines()) if rows.exists() else 0
        report(f'  mirror has {name}: {got}/{want} steps')
        if got < want:
            missing.append(f'{name} {got}/{want}')
    if missing:
        report('  NOT complete: ' + ', '.join(missing))
        return False
    return True


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the Windows console codepage cannot print every glyph
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--remote', default='/data/rlt/2026-09-18-phase2')
    parser.add_argument('--mirror', default='__RLT_ROOT__/vm-mirror')
    parser.add_argument('--interval', type=int, default=600)
    parser.add_argument('--max-hours', type=float, default=72.)
    parser.add_argument('--no-shutdown', action='store_true')
    args = parser.parse_args()

    mirror = Path(args.mirror)
    started = time.time()
    unreachable = 0

    while time.time() - started < args.max_hours * 3600:
        stamp = time.strftime('%H:%M:%S')
        state = pipeline_state(args.key, args.host, args.remote)
        if not state['reachable']:
            unreachable += 1
            print(f'[{stamp}] VM unreachable ({unreachable}) - {state.get("raw", "")}', flush=True)
            if unreachable >= 3:
                print('VM has been unreachable three times running. It may already be gone; the mirror holds '
                      'everything pulled up to the last successful cycle.', flush=True)
                return
            time.sleep(args.interval); continue
        unreachable = 0
        runs = ' '.join(f'{k} {v["steps"]}/{v["total"]}' for k, v in state['runs'].items())
        print(f'[{stamp}] {"running" if state["alive"] else "STOPPED"} | {runs} | M1 done {state["m1_done"]}',
              flush=True)

        if not state['alive']:
            print('pipeline is gone; securing data before touching the machine', flush=True)
            verified = secure_and_verify(args.key, args.host, args.remote, mirror, print)
            if not verified:
                print('DATA NOT VERIFIED - leaving the VM up on purpose. Shutting it down now would take the '
                      'missing data with it. Needs a look.', flush=True)
                return
            print('every expected artefact is mirrored and verified here', flush=True)
            if args.no_shutdown:
                print('--no-shutdown: leaving the machine up'); return
            out = ssh(args.key, args.host, 'sudo shutdown -h now', timeout=30)
            print(f'shutdown issued (rc={out.returncode}); the meter stops here', flush=True)
            return
        time.sleep(args.interval)

    print(f'watchdog reached its {args.max_hours} h limit with the pipeline still running', flush=True)


if __name__ == '__main__':
    main()
